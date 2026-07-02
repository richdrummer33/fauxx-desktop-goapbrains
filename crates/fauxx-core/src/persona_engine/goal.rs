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

use crate::persona::CategoryPool;
use crate::persona_engine::kernel::{BehaviorState, TickContext};
use crate::persona_engine::policy::PersonaPolicy;

/// A concrete goal chosen by the goal layer for one tick.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct DecoyGoal {
    /// A stable-ish id for this goal (persona + routine + category + tick hour).
    pub id: String,
    /// The persona policy id.
    pub persona_id: String,
    /// The routine that produced this goal.
    pub routine: String,
    /// The goal-type label (e.g. `continue_hobby_thread`).
    pub goal_type: String,
    /// The action type (MVP: always `search`; page_visit rides as a module).
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

/// Weight on continuing an existing topic thread (momentum).
const MOMENTUM_WEIGHT: f64 = 1.0;
/// Weight on the novelty bonus (scaled by curiosity).
const NOVELTY_WEIGHT: f64 = 0.6;
/// Penalty applied to a category currently on cooldown.
const COOLDOWN_PENALTY: f64 = 5.0;

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

        // Candidate categories: the routine's categories intersected with the
        // policy's allowed set; fall back to the whole allowed set.
        let routine = policy.routines.iter().find(|r| r.name == routine_name);
        let allowed = policy.allowed_category_pool();
        let mut candidates: Vec<CategoryPool> = match routine {
            Some(r) if !r.categories.is_empty() => r
                .categories
                .iter()
                .filter_map(|n| CategoryPool::from_name(n))
                .filter(|c| allowed.contains(c))
                .collect(),
            _ => allowed.clone(),
        };
        if candidates.is_empty() {
            candidates = allowed;
        }
        if candidates.is_empty() {
            return GoalSelection {
                goal: None,
                scores: Vec::new(),
                idle: true,
                note: "no allowed categories to pursue".to_string(),
            };
        }

        // Score each candidate.
        let max_momentum = candidates
            .iter()
            .map(|c| state.topic_score(c.as_name()))
            .fold(0.0_f64, f64::max)
            .max(1.0);
        let mut scores: Vec<(CategoryPool, f64)> = candidates
            .iter()
            .map(|&c| {
                let momentum = state.topic_score(c.as_name());
                let novelty = (1.0 - momentum / max_momentum).clamp(0.0, 1.0) * state.curiosity;
                let cooldown = if state.on_cooldown(&format!("category:{}", c.as_name()), now) {
                    COOLDOWN_PENALTY
                } else {
                    0.0
                };
                let score = MOMENTUM_WEIGHT * momentum + NOVELTY_WEIGHT * novelty - cooldown;
                (c, score)
            })
            .collect();

        // A boring pivot: with some probability pick the LOWEST (non-cooldown)
        // scorer instead of the highest. Models "boring pivots, mild
        // contradictions" so the persona is not perfectly coherent.
        let pivot = rng.random::<f64>() < policy.topic_decay.pivot_probability;
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let chosen_category = if pivot {
            // Lowest scorer that is not on cooldown, else the top one.
            scores
                .iter()
                .rev()
                .find(|(c, _)| !state.on_cooldown(&format!("category:{}", c.as_name()), now))
                .or_else(|| scores.first())
                .map(|(c, _)| *c)
        } else {
            scores.first().map(|(c, _)| *c)
        };
        let Some(category) = chosen_category else {
            return GoalSelection {
                goal: None,
                scores: score_labels(&scores),
                idle: true,
                note: "every candidate category is on cooldown".to_string(),
            };
        };

        // Subcategory: a topic seed for the chosen category (rng pick).
        let seeds = policy.topic_seeds_for(category);
        let subcategory = if seeds.is_empty() {
            None
        } else {
            Some(seeds[rng.random_range(0..seeds.len())].clone())
        };

        // Goal type from the routine (rng pick), or a sensible default.
        let goal_type = match routine {
            Some(r) if !r.goal_types.is_empty() => {
                r.goal_types[rng.random_range(0..r.goal_types.len())].clone()
            }
            _ => "browse_topic".to_string(),
        };

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
            scores: score_labels(&scores),
            idle: false,
            note: "goal selected".to_string(),
        }
    }
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
    fn evening_selects_an_allowed_category_goal() {
        let policy = elias();
        let kernel = BehaviorKernel;
        let mut state = BehaviorState::new("elias_rickensworth");
        let t = ts(4, 20);
        let tick = kernel.advance(&mut state, &policy, t);
        // Seed chosen so the skip-day draw does not fire; try a few if needed.
        for seed in 0..50u64 {
            let mut rng = StdRng::seed_from_u64(seed);
            let sel = GoalLayer.select(&policy, &state, &tick, t, &mut rng);
            if let Some(goal) = sel.goal {
                assert_eq!(goal.routine, "weekday_evening");
                // The chosen category is one the policy allows.
                assert!(policy.allowed_categories.contains(&goal.category));
                // Evening favors CRAFTS/HISTORY/OUTDOOR_RECREATION.
                assert!(
                    ["CRAFTS", "HISTORY", "OUTDOOR_RECREATION"].contains(&goal.category.as_str())
                );
                assert_eq!(goal.action_type, "search");
                assert!(goal.budget.max_actions >= 1);
                assert!(!goal.reason.is_empty());
                return;
            }
        }
        panic!("expected at least one non-idle selection across seeds");
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
    fn cooldown_category_is_penalized_out_of_selection() {
        let policy = elias();
        let kernel = BehaviorKernel;
        let mut state = BehaviorState::new("elias_rickensworth");
        let t = ts(4, 20);
        let tick = kernel.advance(&mut state, &policy, t);
        // Put every evening category except CRAFTS is not possible; instead put
        // CRAFTS and HISTORY on cooldown and confirm the survivor is chosen.
        state
            .cooldowns
            .insert("category:CRAFTS".to_string(), t + 10_000_000);
        state
            .cooldowns
            .insert("category:HISTORY".to_string(), t + 10_000_000);
        for seed in 0..50u64 {
            let mut rng = StdRng::seed_from_u64(seed);
            let sel = GoalLayer.select(&policy, &state, &tick, t, &mut rng);
            if let Some(goal) = sel.goal {
                // A non-pivot pick must avoid the cooled categories; a pivot also
                // prefers non-cooldown, so OUTDOOR_RECREATION is the expected pick.
                assert_eq!(goal.category, "OUTDOOR_RECREATION");
                return;
            }
        }
        panic!("expected a non-idle selection");
    }
}
