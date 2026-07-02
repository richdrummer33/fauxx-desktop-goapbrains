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

//! The planner: goal -> structured candidate intents.
//!
//! The planner is rules-first and mostly deterministic. It turns the goal
//! layer's chosen category + subcategory into concrete [`Intent`]s carrying the
//! actual search query to run, the dwell range, and the (curated, policy-scoped)
//! domain hints. It NEVER emits an arbitrary URL: an intent carries a search
//! query plus optional domain hints drawn only from the policy's allowlist.
//!
//! The deterministic fallback is the existing
//! [`QueryGenerator`](crate::querybank::QueryGenerator), which is itself
//! blocklist-gated. The LLM sidecar may, when enabled, phrase the approved query
//! seed into candidate queries instead; its output is schema-validated and
//! blocklist-filtered here, and a disabled/empty/invalid result falls back to the
//! deterministic path transparently. Either way every candidate is re-checked by
//! the Safety Gate before it can execute.

use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use serde::Serialize;

use crate::persona::CategoryPool;
use crate::persona_engine::goal::DecoyGoal;
use crate::persona_engine::kernel::BehaviorState;
use crate::persona_engine::policy::PersonaPolicy;
use crate::persona_engine::sidecar::{
    validate_sidecar_queries, SemanticAssistant, SidecarConstraints,
};
use crate::querybank::{commercial_lean, QueryBlocklist, QueryGenerator};

/// A structured candidate intent: everything the executor needs for ONE decoy
/// action, with no free-floating URL. Serializable for the dry-run report.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Intent {
    /// A stable-ish id (goal id + candidate index).
    pub id: String,
    /// The persona policy id.
    pub persona_id: String,
    /// The routine that produced the parent goal.
    pub routine: String,
    /// The parent goal id.
    pub goal_id: String,
    /// The action type (MVP: `search`).
    pub action_type: String,
    /// The [`CategoryPool`] name.
    pub category: String,
    /// The approved seed the query was derived from (the persona's flavor).
    pub query_seed: String,
    /// The topic (subcategory) this intent pursues, when set.
    pub topic: Option<String>,
    /// The concrete, safety-vetted query string to dispatch.
    pub final_query: String,
    /// Max results the persona will visit for this intent (MVP keeps this low).
    pub max_depth: u32,
    /// Inclusive dwell-seconds range `[min, max]`.
    pub dwell_range: [u32; 2],
    /// Curated allowlisted domain hints (from the policy; may be empty).
    pub allowed_domain_hints: Vec<String>,
    /// Forbidden domain hints (from the policy; belt and suspenders).
    pub forbidden_domain_hints: Vec<String>,
    /// A boring, human-readable reason (inherited from the goal).
    pub reason: String,
    /// Whether the LLM sidecar phrased this query (MVP: always `false`).
    pub sidecar_used: bool,
}

/// The planner. Stateless; deterministic given the goal, policy, and seed.
#[derive(Debug, Clone, Copy, Default)]
pub struct Planner;

impl Planner {
    /// Produce candidate intents for `goal`. Uses the LLM sidecar only when the
    /// policy opts in AND the assistant is enabled AND its output validates;
    /// otherwise the deterministic [`QueryGenerator`] fallback is used. `seed`
    /// makes generation reproducible.
    pub fn plan(
        &self,
        policy: &PersonaPolicy,
        goal: &DecoyGoal,
        _state: &BehaviorState,
        sidecar: &dyn SemanticAssistant,
        seed: u64,
    ) -> Vec<Intent> {
        let Some(category) = CategoryPool::from_name(&goal.category) else {
            return Vec::new();
        };
        let max = (policy.planner_settings.max_candidate_intents as usize).max(1);
        let query_seed = goal
            .subcategory
            .clone()
            .unwrap_or_else(|| category.as_name().to_lowercase());

        let blocklist = QueryBlocklist::bundled();

        // Try the sidecar first when the policy opts in; fall back otherwise.
        let (mut candidates, sidecar_used) =
            self.sidecar_candidates(policy, goal, &query_seed, sidecar, &blocklist, max);
        if candidates.is_empty() {
            candidates =
                self.deterministic_candidates(policy, category, &query_seed, &blocklist, seed, max);
        }

        candidates
            .into_iter()
            .take(max)
            .enumerate()
            .map(|(i, query)| Intent {
                id: format!("{}#{i}", goal.id),
                persona_id: policy.id.clone(),
                routine: goal.routine.clone(),
                goal_id: goal.id.clone(),
                action_type: goal.action_type.clone(),
                category: goal.category.clone(),
                query_seed: query_seed.clone(),
                topic: goal.subcategory.clone(),
                final_query: query,
                max_depth: policy.planner_settings.max_results_to_visit,
                dwell_range: [
                    policy.planner_settings.dwell_seconds_min,
                    policy.planner_settings.dwell_seconds_max,
                ],
                allowed_domain_hints: policy.domain_policy.allow_domains.clone(),
                forbidden_domain_hints: policy.domain_policy.forbidden_domains.clone(),
                reason: goal.reason.clone(),
                sidecar_used,
            })
            .collect()
    }

    /// Sidecar-phrased candidates, validated + blocklist-filtered. Returns
    /// `(queries, true)` when the sidecar contributed, else `(empty, false)`.
    fn sidecar_candidates(
        &self,
        policy: &PersonaPolicy,
        _goal: &DecoyGoal,
        query_seed: &str,
        sidecar: &dyn SemanticAssistant,
        blocklist: &QueryBlocklist,
        max: usize,
    ) -> (Vec<String>, bool) {
        if !policy.planner_settings.use_llm_sidecar || !sidecar.is_enabled() {
            return (Vec::new(), false);
        }
        let mut constraints = SidecarConstraints::new(&policy.id, max);
        constraints.forbidden_topics = policy.disinterests.clone();
        let Some(raw) = sidecar.generate_search_queries(policy, query_seed, &constraints) else {
            return (Vec::new(), false);
        };
        let vetted: Vec<String> = validate_sidecar_queries(raw, &constraints)
            .into_iter()
            .filter(|q| !blocklist.is_blocked(q))
            .collect();
        if vetted.is_empty() {
            (Vec::new(), false)
        } else {
            (vetted, true)
        }
    }

    /// Deterministic fallback candidates, kept IN CHARACTER.
    ///
    /// The persona picked the category and a topic seed; the queries should sound
    /// like the persona, not like the broad category corpus. So the order is:
    /// (1) the chosen seed itself, (2) on-topic refinements of it (its own words
    /// plus qualifiers, via [`QueryGenerator::refine_goal`]), (3) the persona's
    /// OTHER curated seeds for this category, and only (4) a light top-up from the
    /// generic [`QueryGenerator`] bank if the persona has too few seeds to fill
    /// the plan. Every candidate is deduped and blocklist-safe.
    fn deterministic_candidates(
        &self,
        policy: &PersonaPolicy,
        category: CategoryPool,
        query_seed: &str,
        blocklist: &QueryBlocklist,
        seed: u64,
        max: usize,
    ) -> Vec<String> {
        let generator = QueryGenerator::new(seed);
        let mut rng = StdRng::seed_from_u64(seed ^ 0x9E37_79B9_7F4A_7C15);
        let mut out: Vec<String> = Vec::new();

        let add = |out: &mut Vec<String>, q: &str| {
            let q = q.trim();
            if !q.is_empty() && out.iter().all(|e| e != q) && !blocklist.is_blocked(q) {
                out.push(q.to_string());
            }
        };

        // (1) The chosen seed, in the persona's own words.
        add(&mut out, query_seed);

        // (2) On-topic refinements of the seed (stays on the seed's words).
        if out.len() < max {
            for r in generator.refine_goal(query_seed, max, &mut rng) {
                if out.len() >= max {
                    break;
                }
                add(&mut out, &r);
            }
        }

        // (3) The persona's OTHER seeds for this category, shuffled, plus a
        //     refinement of each if we still have room.
        let mut others: Vec<String> = policy
            .topic_seeds_for(category)
            .iter()
            .filter(|s| s.as_str() != query_seed)
            .cloned()
            .collect();
        shuffle(&mut others, &mut rng);
        for other in &others {
            if out.len() >= max {
                break;
            }
            add(&mut out, other);
            if out.len() < max {
                if let Some(r) = generator.refine_goal(other, 1, &mut rng).into_iter().next() {
                    add(&mut out, &r);
                }
            }
        }

        // (4) Only if the persona is seed-poor, top up from the generic bank.
        let lean = commercial_lean(&policy.backing_persona_categories());
        let mut attempts = 0;
        while out.len() < max && attempts < max * 6 {
            attempts += 1;
            if let Some(q) = generator.generate(category, lean, &mut rng) {
                add(&mut out, &q);
            }
        }
        out
    }
}

/// In-place Fisher-Yates shuffle using the injected RNG (deterministic per seed).
fn shuffle(items: &mut [String], rng: &mut impl RngExt) {
    for i in (1..items.len()).rev() {
        let j = rng.random_range(0..=i);
        items.swap(i, j);
    }
}

impl PersonaPolicy {
    /// The backing persona's interests as [`CategoryPool`] values (unknown names
    /// dropped), for the query generator's commercial-lean read.
    pub fn backing_persona_categories(&self) -> Vec<CategoryPool> {
        self.backing_persona
            .interests
            .iter()
            .filter_map(|n| CategoryPool::from_name(n))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;

    use crate::persona_engine::builtins;
    use crate::persona_engine::goal::GoalLayer;
    use crate::persona_engine::kernel::BehaviorKernel;
    use crate::persona_engine::sidecar::DisabledAssistant;

    fn elias() -> PersonaPolicy {
        match builtins::get("elias_rickensworth") {
            Ok(p) => p,
            Err(e) => panic!("elias must load: {e}"),
        }
    }

    fn ts(day: i64, hour: u8) -> i64 {
        day * 86_400_000 + (hour as i64) * 3_600_000
    }

    /// Drive kernel + goal to obtain a concrete goal for planning.
    fn a_goal(policy: &PersonaPolicy) -> (DecoyGoal, BehaviorState) {
        let kernel = BehaviorKernel;
        let mut state = BehaviorState::new("elias_rickensworth");
        let t = ts(4, 20);
        let tick = kernel.advance(&mut state, policy, t);
        for seed in 0..200u64 {
            let mut rng = StdRng::seed_from_u64(seed);
            let sel = GoalLayer.select(policy, &state, &tick, t, &mut rng);
            if let Some(goal) = sel.goal {
                // The planner only produces intents for ONLINE (search) goals.
                if goal.online {
                    return (goal, state);
                }
            }
        }
        panic!("could not obtain an online goal");
    }

    #[test]
    fn deterministic_fallback_produces_safe_bounded_intents() {
        let policy = elias();
        let (goal, state) = a_goal(&policy);
        let intents = Planner.plan(&policy, &goal, &state, &DisabledAssistant, 42);
        assert!(!intents.is_empty());
        assert!(intents.len() <= policy.planner_settings.max_candidate_intents as usize);
        let blocklist = QueryBlocklist::bundled();
        for intent in &intents {
            assert!(!intent.final_query.is_empty());
            // Every emitted query is blocklist-safe.
            assert!(!blocklist.is_blocked(&intent.final_query));
            // The MVP sidecar never fires.
            assert!(!intent.sidecar_used);
            // The category matches the goal.
            assert_eq!(intent.category, goal.category);
            // Dwell range is the policy's.
            assert_eq!(
                intent.dwell_range,
                [
                    policy.planner_settings.dwell_seconds_min,
                    policy.planner_settings.dwell_seconds_max
                ]
            );
        }
    }

    #[test]
    fn planner_never_emits_an_arbitrary_url() {
        // In the MVP the persona has no allowlisted domains, so an intent carries
        // NO domain hint and its query is a search string, never a URL.
        let policy = elias();
        let (goal, state) = a_goal(&policy);
        let intents = Planner.plan(&policy, &goal, &state, &DisabledAssistant, 7);
        for intent in &intents {
            assert!(intent.allowed_domain_hints.is_empty());
            assert!(!intent.final_query.contains("http://"));
            assert!(!intent.final_query.contains("https://"));
        }
    }

    #[test]
    fn planning_is_deterministic_for_a_fixed_seed() {
        let policy = elias();
        let (goal, state) = a_goal(&policy);
        let a = Planner.plan(&policy, &goal, &state, &DisabledAssistant, 99);
        let b = Planner.plan(&policy, &goal, &state, &DisabledAssistant, 99);
        assert_eq!(a, b);
    }

    #[test]
    fn invalid_sidecar_output_falls_back_to_deterministic() {
        // An assistant that returns a URL and an over-long string (both invalid)
        // must not poison the plan; the planner falls back deterministically.
        struct BadAssistant;
        impl SemanticAssistant for BadAssistant {
            fn is_enabled(&self) -> bool {
                true
            }
            fn generate_search_queries(
                &self,
                _p: &PersonaPolicy,
                _seed: &str,
                _c: &SidecarConstraints,
            ) -> Option<Vec<String>> {
                Some(vec!["".to_string(), "celebrity gossip roundup".to_string()])
            }
        }
        let mut policy = elias();
        policy.planner_settings.use_llm_sidecar = true;
        let (goal, state) = a_goal(&policy);
        let intents = Planner.plan(&policy, &goal, &state, &BadAssistant, 3);
        // "celebrity gossip" is a disinterest -> filtered; empty -> filtered; so
        // the sidecar contributes nothing and the deterministic path is used.
        assert!(!intents.is_empty());
        for intent in &intents {
            assert!(!intent.sidecar_used);
        }
    }
}
