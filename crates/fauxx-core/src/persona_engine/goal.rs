// fauxx-desktop: Fauxx Desktop Companion
// Copyright (C) 2026 Digital Grease
//
// This program is free software: you can redistribute it and/or modify it
// under the terms of the GNU Affero General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version.
//
// This program is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more
// details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! The goal layer: the control plane.
//!
//! Given the persona policy and the current [`BehaviorState`], the goal layer
//! chooses ONE concrete goal for this tick: a routine, a goal type, an action
//! type, a category, a subcategory (topic seed), a bounded budget, and the
//! allowed modules / forbidden capabilities. This is where "what does the
//! persona want right now" is decided, DETERMINISTICALLY. The LLM has no say
//! here.
//!
//! Selection is a small utility model over the routine's candidate categories:
//! topic momentum (continue a thread), a novelty bonus scaled by curiosity, and
//! a cooldown penalty. To avoid being too coherent (itself a fingerprint) the
//! persona occasionally takes a boring off-momentum pivot or skips the run
//! entirely; both are drawn from the injected RNG so a fixed seed is
//! reproducible.

use rand::RngExt;
use serde::Serialize;

use std::collections::HashMap;

use crate::persona::CategoryPool;
use crate::persona_engine::kernel::{BehaviorState, TickContext};
use crate::persona_engine::policy::{Domain, PersonaPolicy, Routine};
use crate::persona_engine::utility;
use crate::persona_engine::world::{weighted_pick, Interest};

/// A concrete goal chosen by the goal layer for one tick.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct DecoyGoal {
    /// A stable-ish id for this goal (persona + routine + category + tick hour).
    pub id: String,
    /// The persona policy id.
    pub persona_id: String,
    /// The routine that produced this goal.
    pub routine: String,
    /// The need/motive (life domain) this goal services (e.g. `hobby`, `upkeep`).
    pub need: String,
    /// Whether this goal runs ONLINE (a decoy search) or OFFLINE (a real-world
    /// errand that emits nothing on the wire).
    pub online: bool,
    /// The goal-type label (e.g. `continue_hobby_thread`).
    pub goal_type: String,
    /// The action type: `search` for online, `offline` for a real-world errand.
    pub action_type: String,
    /// The chosen [`CategoryPool`] name (drives the real query generator).
    pub category: String,
    /// The chosen topic seed / subcategory, when the policy declares one.
    pub subcategory: Option<String>,
    /// The bounded budget for this goal.
    pub budget: GoalBudget,
    /// The executor modules this goal may use (from the policy).
    pub allowed_modules: Vec<String>,
    /// The capabilities this goal must never exercise (from the policy).
    pub forbidden_capabilities: Vec<String>,
    /// A boring, human-readable reason (the goal layer writes this, not the LLM).
    pub reason: String,
}

/// A goal's bounded budget, copied from the policy's action budget.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct GoalBudget {
    /// Max decoy actions this goal may perform.
    pub max_actions: u32,
    /// Max wall-clock minutes this goal may span.
    pub max_minutes: u32,
}

/// The full output of a goal-selection pass: the chosen goal (absent when idle),
/// the per-category utility scores, and whether the tick is idle.
#[derive(Debug, Clone)]
pub struct GoalSelection {
    /// The selected goal, or `None` when idle (quiet hours or a skipped day).
    pub goal: Option<DecoyGoal>,
    /// The `(category, score)` pairs the utility model computed, for the report.
    pub scores: Vec<(String, f64)>,
    /// Whether this tick is idle.
    pub idle: bool,
    /// A human-readable note on why (quiet hours, skipped day, or selected).
    pub note: String,
}

/// Base softmax temperature for category selection. The policy's
/// `pivot_probability` nudges it up, so a more restless persona explores more.
const BASE_TEMPERATURE: f64 = 0.25;

/// The goal layer. Stateless; reads the policy + state and chooses a goal.
#[derive(Debug, Clone, Copy, Default)]
pub struct GoalLayer;

impl GoalLayer {
    /// Select a goal for this tick. `now` is epoch millis (for cooldown checks);
    /// `rng` supplies the pivot / skip-day / seed choices so a fixed seed is
    /// reproducible.
    pub fn select(
        &self,
        policy: &PersonaPolicy,
        state: &BehaviorState,
        tick: &TickContext,
        now: i64,
        rng: &mut impl RngExt,
    ) -> GoalSelection {
        // Quiet hours: nothing to do.
        let Some(routine_name) = tick.routine.clone() else {
            return GoalSelection {
                goal: None,
                scores: Vec::new(),
                idle: true,
                note: "quiet hours: no routine covers this time".to_string(),
            };
        };

        // Occasionally skip the whole run (an idle day) to avoid being too
        // regular. Drawn first so it does not depend on the category scoring.
        if rng.random::<f64>() < policy.topic_decay.skip_day_probability {
            return GoalSelection {
                goal: None,
                scores: Vec::new(),
                idle: true,
                note: "skipped run (anti-coherence idle draw)".to_string(),
            };
        }

        let routine = policy.routines.iter().find(|r| r.name == routine_name);
        let temperature = BASE_TEMPERATURE + policy.topic_decay.pivot_probability;

        // --- DESIRE: choose which NEED (life domain) to service this session ---
        // The most-deficient need pulls hardest, weighted by how well the domain
        // fits the current routine; drawn with a temperature softmax.
        let domains = policy.effective_domains();
        let desires: Vec<f64> = domains
            .iter()
            .map(|d| domain_desire(d, routine, state))
            .collect();
        let Some(domain_idx) = utility::softmax_choose(&desires, temperature, rng) else {
            return GoalSelection {
                goal: None,
                scores: Vec::new(),
                idle: true,
                note: "no life domain to service".to_string(),
            };
        };
        let domain = &domains[domain_idx];

        // --- Decide ONLINE (decoy search) vs OFFLINE (real-world, no traffic) ---
        let online_candidates: Vec<CategoryPool> = domain
            .categories
            .iter()
            .filter_map(|n| CategoryPool::from_name(n))
            .filter(|c| policy.allowed_categories.iter().any(|a| a == c.as_name()))
            .collect();
        let go_online = !online_candidates.is_empty() && rng.random::<f64>() < domain.online_bias;

        if !go_online {
            // Offline action: attend to the need in the real world. Nothing goes
            // on the wire, which is exactly how a sensitive need (health, errands)
            // is served WITHOUT ever becoming decoy query signal.
            return GoalSelection {
                goal: Some(offline_goal(policy, domain, &routine_name, tick.hour)),
                scores: Vec::new(),
                idle: false,
                note: format!("offline: servicing '{}' in the real world", domain.name),
            };
        }

        // --- GOAL: pick a category within the domain via the utility model ---
        let Some((scored, chosen_idx)) =
            pick_category(policy, &online_candidates, state, now, rng, temperature)
        else {
            // Everything the domain could search is on cooldown; do it offline
            // rather than forcing a repeat.
            return GoalSelection {
                goal: Some(offline_goal(policy, domain, &routine_name, tick.hour)),
                scores: Vec::new(),
                idle: false,
                note: format!("offline fallback: '{}' categories on cooldown", domain.name),
            };
        };
        let category = scored[chosen_idx].0;
        let raw: Vec<f64> = scored.iter().map(|(_, s)| *s).collect();
        // A "pivot" is just when the sampler did not land on the top scorer;
        // used only to flavor the reason.
        let pivot = utility::argmax(&raw) != Some(chosen_idx);

        // Subcategory: walk THROUGH the persona's seeds for this category,
        // preferring one not used recently (interest threading), so consecutive
        // sessions progress (ink -> blotting paper) instead of repeating one.
        let subcategory = choose_seed(policy, category, state, rng);
        let goal_type = pick_goal_type(domain, routine, rng);
        let reason = build_reason(
            policy,
            &routine_name,
            category,
            subcategory.as_deref(),
            pivot,
        );

        let goal = DecoyGoal {
            id: format!(
                "{}:{}:{}:{}",
                policy.id,
                routine_name,
                category.as_name(),
                tick.hour
            ),
            persona_id: policy.id.clone(),
            routine: routine_name,
            need: domain.name.clone(),
            online: true,
            goal_type,
            action_type: "search".to_string(),
            category: category.as_name().to_string(),
            subcategory,
            budget: GoalBudget {
                max_actions: policy.action_budget.max_actions_per_run,
                max_minutes: policy.action_budget.max_minutes_per_run,
            },
            allowed_modules: policy.safety_policy.allowed_modules.clone(),
            forbidden_capabilities: policy.safety_policy.forbidden_capabilities.clone(),
            reason,
        };

        GoalSelection {
            goal: Some(goal),
            scores: score_labels(&scored),
            idle: false,
            note: "goal selected".to_string(),
        }
    }
}

/// The desire to service a domain's need this session: its deficit, weighted by
/// how well the domain fits the current routine. Offline-only domains stay
/// somewhat eligible at any time (real life does not keep a schedule).
fn domain_desire(domain: &Domain, routine: Option<&Routine>, state: &BehaviorState) -> f64 {
    let deficit = state.needs.deficit(&domain.name);
    let fit = if domain.categories.is_empty() {
        0.6
    } else if routine.is_some_and(|r| {
        domain
            .categories
            .iter()
            .any(|c| r.categories.iter().any(|rc| rc == c))
    }) {
        1.0
    } else {
        0.5
    };
    fit * (0.2 + deficit)
}

/// Score the domain's candidate categories with the utility model and draw one
/// with a temperature softmax. `None` when every candidate is on cooldown.
fn pick_category(
    policy: &PersonaPolicy,
    candidates: &[CategoryPool],
    state: &BehaviorState,
    now: i64,
    rng: &mut impl RngExt,
    temperature: f64,
) -> Option<(Vec<(CategoryPool, f64)>, usize)> {
    if candidates.is_empty() {
        return None;
    }
    let max_momentum = candidates
        .iter()
        .map(|c| state.topic_score(c.as_name()))
        .fold(0.0_f64, f64::max)
        .max(1.0);
    let affinity = routine_affinity(policy);
    let recent_len = state.recent_topics.len().max(1) as f64;
    let scored: Vec<(CategoryPool, f64)> = candidates
        .iter()
        .map(|&c| {
            let name = c.as_name();
            let momentum = state.topic_score(name) / max_momentum;
            let novelty = (1.0 - momentum).clamp(0.0, 1.0) * state.curiosity;
            let recency = state
                .recent_topics
                .iter()
                .filter(|t| t.as_str() == name)
                .count() as f64
                / recent_len;
            let cons = utility::Considerations {
                affinity: affinity.get(name).copied().unwrap_or(0.5),
                momentum,
                novelty,
                recency,
                on_cooldown: state.on_cooldown(&format!("category:{name}"), now),
            };
            (c, utility::score(&cons))
        })
        .collect();
    let raw: Vec<f64> = scored.iter().map(|(_, s)| *s).collect();
    let idx = utility::softmax_choose(&raw, temperature, rng)?;
    Some((scored, idx))
}

/// Build an OFFLINE goal for a domain: a real-world errand that emits nothing on
/// the wire but still services (and later satisfies) the need.
fn offline_goal(
    policy: &PersonaPolicy,
    domain: &Domain,
    routine_name: &str,
    hour: u8,
) -> DecoyGoal {
    let label = domain
        .offline_label
        .clone()
        .unwrap_or_else(|| format!("attends to {}", domain.name));
    DecoyGoal {
        id: format!("{}:{}:{}:{}", policy.id, routine_name, domain.name, hour),
        persona_id: policy.id.clone(),
        routine: routine_name.to_string(),
        need: domain.name.clone(),
        online: false,
        goal_type: domain
            .goal_types
            .first()
            .cloned()
            .unwrap_or_else(|| "offline_errand".to_string()),
        action_type: "offline".to_string(),
        category: String::new(),
        subcategory: Some(label.clone()),
        budget: GoalBudget {
            max_actions: 1,
            max_minutes: policy.action_budget.max_minutes_per_run,
        },
        allowed_modules: policy.safety_policy.allowed_modules.clone(),
        forbidden_capabilities: policy.safety_policy.forbidden_capabilities.clone(),
        reason: format!(
            "{} {label} (offline; nothing on the wire)",
            policy.display_name
        ),
    }
}

/// Pick a goal-type label: prefer the domain's, else the routine's, else a default.
fn pick_goal_type(domain: &Domain, routine: Option<&Routine>, rng: &mut impl RngExt) -> String {
    if !domain.goal_types.is_empty() {
        return domain.goal_types[rng.random_range(0..domain.goal_types.len())].clone();
    }
    match routine {
        Some(r) if !r.goal_types.is_empty() => {
            r.goal_types[rng.random_range(0..r.goal_types.len())].clone()
        }
        _ => "browse_topic".to_string(),
    }
}

/// Per-category affinity in `[0.4, 1.0]`, from how many routines favor a category
/// (a proxy for how CORE it is to the persona): a category named by every routine
/// scores 1.0, a one-routine category floors at 0.4. Keyed by category name.
fn routine_affinity(policy: &PersonaPolicy) -> HashMap<String, f64> {
    let mut counts: HashMap<String, usize> = HashMap::new();
    for c in policy.allowed_category_pool() {
        let name = c.as_name().to_string();
        let n = policy
            .routines
            .iter()
            .filter(|r| r.categories.iter().any(|rc| rc == &name))
            .count();
        counts.insert(name, n);
    }
    let max = counts.values().copied().max().unwrap_or(0).max(1) as f64;
    counts
        .into_iter()
        .map(|(name, n)| (name, 0.4 + 0.6 * (n as f64 / max)))
        .collect()
}

/// Choose a topic seed for `category`, threading through the persona's LIVING
/// interest graph ([`crate::persona_engine::world::WorldState`]), not a flat
/// authored list. This is what lets a discovered interest (adopted by
/// appraisal) start pulling seed selection once it has been reinforced: the
/// graph, not the policy, is now the source of truth for "what does he care
/// about right now."
///
/// Priority: (1) an AUTHORED follow-up of the most recent seed that lives in
/// this category (so an arc unfolds: ink -> blotting paper -> nib grinding),
/// weighted-picked among the arc candidates; else (2) a not-recently-used
/// candidate, weighted by [`Interest::effective_weight`] (so a hot authored
/// interest or a reinforced discovery pulls harder); else (3) any candidate in
/// the category, same weighting. `None` when the category has no interest nodes
/// (the graph is always seeded from `topic_seeds` before this runs).
fn choose_seed(
    policy: &PersonaPolicy,
    category: CategoryPool,
    state: &BehaviorState,
    rng: &mut impl RngExt,
) -> Option<String> {
    let candidates = state.world.interests_for_category(category.as_name());
    if candidates.is_empty() {
        return None;
    }

    // (1) Authored narrative arc: follow the most recent seed's declared
    // follow-ups, restricted to this category's candidates and not-just-used.
    if let Some(last) = state.recent_seeds.last() {
        if let Some(followups) = policy.seed_followups.get(last) {
            let arc: Vec<&Interest> = candidates
                .iter()
                .filter(|i| followups.contains(&i.seed) && !state.recent_seeds.contains(&i.seed))
                .copied()
                .collect();
            if let Some(pick) = weighted_pick(&arc, rng) {
                return Some(pick.seed.clone());
            }
        }
    }

    // (2) Otherwise prefer a candidate not used recently; (3) fall back to all
    // candidates in the category. Either way, weighted by how much the persona
    // cares about it right now.
    let fresh: Vec<&Interest> = candidates
        .iter()
        .filter(|i| !state.recent_seeds.contains(&i.seed))
        .copied()
        .collect();
    let pool: Vec<&Interest> = if fresh.is_empty() { candidates } else { fresh };
    weighted_pick(&pool, rng).map(|i| i.seed.clone())
}

/// Convert scored categories to `(name, score)` label pairs for the report.
fn score_labels(scores: &[(CategoryPool, f64)]) -> Vec<(String, f64)> {
    scores
        .iter()
        .map(|(c, s)| (c.as_name().to_string(), *s))
        .collect()
}

/// Build the boring human-readable reason string.
fn build_reason(
    policy: &PersonaPolicy,
    routine: &str,
    category: CategoryPool,
    subcategory: Option<&str>,
    pivot: bool,
) -> String {
    let who = &policy.display_name;
    let topic = subcategory.unwrap_or_else(|| category.as_name());
    if pivot {
        format!(
            "{who} is in the {routine} routine and idly pivots to {topic} ({}).",
            category.as_name()
        )
    } else {
        format!(
            "{who} is in the {routine} routine and continues a {} thread about {topic}.",
            category.as_name()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    use crate::persona_engine::builtins;
    use crate::persona_engine::kernel::BehaviorKernel;

    fn elias() -> PersonaPolicy {
        match builtins::get("elias_rickensworth") {
            Ok(p) => p,
            Err(e) => panic!("elias must load: {e}"),
        }
    }

    fn ts(day: i64, hour: u8) -> i64 {
        day * 86_400_000 + (hour as i64) * 3_600_000
    }

    #[test]
    fn quiet_hours_yield_no_goal() {
        let policy = elias();
        let kernel = BehaviorKernel;
        let mut state = BehaviorState::new("elias_rickensworth");
        let tick = kernel.advance(&mut state, &policy, ts(4, 3));
        let mut rng = StdRng::seed_from_u64(1);
        let sel = GoalLayer.select(&policy, &state, &tick, ts(4, 3), &mut rng);
        assert!(sel.idle);
        assert!(sel.goal.is_none());
    }

    #[test]
    fn evening_yields_an_online_hobby_search() {
        let policy = elias();
        let kernel = BehaviorKernel;
        let mut state = BehaviorState::new("elias_rickensworth");
        let t = ts(4, 20);
        let tick = kernel.advance(&mut state, &policy, t);
        for seed in 0..80u64 {
            let mut rng = StdRng::seed_from_u64(seed);
            if let Some(goal) = GoalLayer.select(&policy, &state, &tick, t, &mut rng).goal {
                if goal.online {
                    assert_eq!(goal.routine, "weekday_evening");
                    assert_eq!(goal.action_type, "search");
                    assert!(policy.allowed_categories.contains(&goal.category));
                    assert!(!goal.need.is_empty());
                    assert!(!goal.reason.is_empty());
                    return;
                }
            }
        }
        panic!("expected at least one online selection across seeds");
    }

    #[test]
    fn both_online_and_offline_goals_occur() {
        // Elias has online (hobby, upkeep) and offline (wellbeing, errands)
        // domains, so across seeds the goal layer produces both kinds.
        let policy = elias();
        let kernel = BehaviorKernel;
        let mut state = BehaviorState::new("elias_rickensworth");
        let t = ts(2, 11); // Saturday late morning (the weekend routine)
        let tick = kernel.advance(&mut state, &policy, t);
        let mut saw_online = false;
        let mut saw_offline = false;
        for seed in 0..200u64 {
            let mut rng = StdRng::seed_from_u64(seed);
            if let Some(goal) = GoalLayer.select(&policy, &state, &tick, t, &mut rng).goal {
                if goal.online {
                    saw_online = true;
                    assert_eq!(goal.action_type, "search");
                } else {
                    saw_offline = true;
                    assert_eq!(goal.action_type, "offline");
                    assert!(goal.category.is_empty());
                    assert!(!goal.need.is_empty());
                }
            }
            if saw_online && saw_offline {
                return;
            }
        }
        panic!(
            "expected both online and offline goals (online={saw_online}, offline={saw_offline})"
        );
    }

    #[test]
    fn selection_is_deterministic_for_a_fixed_seed() {
        let policy = elias();
        let kernel = BehaviorKernel;
        let mut state = BehaviorState::new("elias_rickensworth");
        let t = ts(4, 20);
        let tick = kernel.advance(&mut state, &policy, t);
        let mut a = StdRng::seed_from_u64(7);
        let mut b = StdRng::seed_from_u64(7);
        let ga = GoalLayer.select(&policy, &state, &tick, t, &mut a);
        let gb = GoalLayer.select(&policy, &state, &tick, t, &mut b);
        assert_eq!(ga.goal, gb.goal);
    }

    #[test]
    fn choose_seed_follows_authored_arc() {
        // Having just pursued "fountain pens", the next CRAFTS seed should be one
        // of its authored follow-ups (an unfolding narrative arc).
        let policy = elias();
        let mut state = BehaviorState::new("elias_rickensworth");
        state.world.ensure_seeded(&policy, 0);
        state.recent_seeds = vec!["fountain pens".to_string()];
        for seed in 0..40u64 {
            let mut rng = StdRng::seed_from_u64(seed);
            let Some(pick) = choose_seed(&policy, CategoryPool::CRAFTS, &state, &mut rng) else {
                panic!("CRAFTS has seeds");
            };
            assert!(
                ["blue black ink", "blotting paper"].contains(&pick.as_str()),
                "expected an arc follow-up of fountain pens, got {pick}"
            );
        }
    }

    #[test]
    fn pick_category_avoids_cooldown() {
        let policy = elias();
        let mut state = BehaviorState::new("elias_rickensworth");
        let t = ts(4, 20);
        state
            .cooldowns
            .insert("category:CRAFTS".to_string(), t + 10_000_000);
        state
            .cooldowns
            .insert("category:HISTORY".to_string(), t + 10_000_000);
        let cats = vec![
            CategoryPool::CRAFTS,
            CategoryPool::HISTORY,
            CategoryPool::OUTDOOR_RECREATION,
        ];
        for seed in 0..50u64 {
            let mut rng = StdRng::seed_from_u64(seed);
            let Some((scored, idx)) = pick_category(&policy, &cats, &state, t, &mut rng, 0.4)
            else {
                panic!("a non-cooled category exists");
            };
            assert_eq!(scored[idx].0, CategoryPool::OUTDOOR_RECREATION);
        }
    }
}
