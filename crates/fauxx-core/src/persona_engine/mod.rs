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

//! The persona engine: a small, safety-constrained decoy behavior simulation.
//!
//! This is "The Sims for privacy decoy personas", but bounded. A deterministic
//! goal/control layer drives behavior; the (MVP-disabled) LLM sidecar is only a
//! narrow, schema-validated helper. It is NOT an autonomous LLM browser agent.
//!
//! The pipeline, top to bottom:
//!
//! 1. [`policy::PersonaPolicy`] - who a persona is ALLOWED to be (TOML).
//! 2. [`kernel::BehaviorKernel`] - a Sims-like state model (time, energy,
//!    curiosity, boredom, topic momentum, cooldowns).
//! 3. [`goal::GoalLayer`] - the control plane: state -> a concrete goal + action
//!    type + category + budget + allowed modules.
//! 4. [`planner::Planner`] - goal -> structured candidate [`planner::Intent`]s
//!    (rules-first; deterministic query fallback).
//! 5. [`sidecar::SemanticAssistant`] - an optional, bounded LLM helper. Disabled
//!    by default; never chooses URLs, never bypasses the Safety Gate.
//! 6. [`safety::SafetyGate`] - mandatory deterministic enforcement; fails closed.
//! 7. Executor - reuses the existing isolated-Chromium decoy browser.
//! 8. [`log::ActivityRecord`] - local-only JSONL activity log (decoy-only).
//!
//! The LLM MUST NOT decide what the persona wants, what to run, which modules to
//! use, or where to browse. The goal layer chooses; the LLM, at most, phrases an
//! already-approved query. See the crate `docs/PERSONA_ENGINE.md`.

pub mod appraise;
pub mod builtins;
pub mod goal;
pub mod kernel;
pub mod llm;
pub mod log;
pub mod needs;
pub mod planner;
pub mod policy;
pub mod safety;
pub mod sidecar;
pub mod stimulus;
mod utility;
pub mod world;

pub use appraise::{appraise, Appraisal};
pub use goal::{DecoyGoal, GoalLayer};
pub use kernel::{BehaviorKernel, BehaviorState};
pub use llm::{LlmConfig, LlmTransport, LmStudioAssistant, LmStudioTransport};
pub use log::ActivityRecord;
pub use needs::NeedState;
pub use planner::{Intent, Planner};
pub use policy::{
    Domain, Identity, LifeEvent, PersonaPolicy, Personality, PolicyIssue, POLICY_SCHEMA_VERSION,
};
pub use safety::{SafetyDecision, SafetyGate};
pub use sidecar::{DisabledAssistant, SemanticAssistant, SidecarConstraints};
pub use stimulus::{roll_life_event, Stimulus, StimulusSource};
pub use world::{Interest, InterestSource, Memory, MemoryKind, WorldState};

use rand::rngs::StdRng;
use rand::SeedableRng;
use serde::Serialize;

use crate::persona::SyntheticPersona;

/// A concise, serializable summary of a persona policy for `persona-engine list`.
#[derive(Debug, Clone, Serialize)]
pub struct PolicySummary {
    /// The policy id.
    pub id: String,
    /// The display name.
    pub display_name: String,
    /// Whether this policy is a bundled built-in.
    pub builtin: bool,
    /// The count of routines declared.
    pub routine_count: usize,
    /// The allowed category names.
    pub allowed_categories: Vec<String>,
}

impl PolicySummary {
    /// Build a summary from a policy, marking whether it is a built-in.
    pub fn from_policy(policy: &PersonaPolicy, builtin: bool) -> Self {
        Self {
            id: policy.id.clone(),
            display_name: policy.display_name.clone(),
            builtin,
            routine_count: policy.routines.len(),
            allowed_categories: policy.allowed_categories.clone(),
        }
    }
}

/// The full result of a persona-engine planning pass, printed by
/// `persona-engine plan`/`run-once --dry-run`. It carries every stage's output
/// so the operator can see exactly how a decision was reached, and it is
/// serializable for `--json`.
///
/// A dry-run never drives the decoy browser and never persists anything, but
/// [`no_network`](Self::no_network) is NOT hardcoded `true`: with the optional
/// LLM sidecar enabled, sensing/appraising a stimulus (and, on a day
/// boundary, reflection) MAY make a real loopback HTTP call to the local LLM
/// server. `no_network` reflects that honestly (`!sidecar.is_enabled()`)
/// rather than asserting a guarantee the sidecar can break.
#[derive(Debug, Clone, Serialize)]
pub struct DryRunReport {
    /// The persona policy id.
    pub persona_id: String,
    /// The policy version stamped on the report.
    pub policy_version: u32,
    /// The behavior state after the kernel advanced it for this tick (includes
    /// any world-model change from [`sensed`](Self::sensed)).
    pub behavior_state: BehaviorState,
    /// The offline stimulus rolled and appraised this tick, if any (`None` on a
    /// tick where no life event fired). Present even when NOT noticed, so a
    /// dry-run can show "something happened but he didn't catch it".
    pub sensed: Option<SensedStimulus>,
    /// The current routine name, or `None` during quiet hours.
    pub current_routine: Option<String>,
    /// Whether this tick is an idle/skipped run (quiet hours or a skipped day).
    pub idle: bool,
    /// The per-goal utility scores the goal layer computed (goal label -> score).
    pub goal_scores: Vec<(String, f64)>,
    /// The selected goal (absent when idle).
    pub selected_goal: Option<DecoyGoal>,
    /// The candidate intents the planner produced (empty when idle).
    pub candidate_intents: Vec<Intent>,
    /// Whether the planner's query generation actually used the LLM sidecar
    /// this tick (`false` whenever the sidecar is disabled; also `false` on a
    /// tick where the deterministic fallback was used anyway). This does NOT
    /// cover the sense/appraisal or reflection seams; see
    /// [`no_network`](Self::no_network) for whether ANY sidecar call could
    /// have happened this tick.
    pub sidecar_used: bool,
    /// The Safety Gate decision for each candidate intent.
    pub safety_decisions: Vec<SafetyDecision>,
    /// The final, safety-approved action plan (the intents that would execute).
    pub final_plan: Vec<Intent>,
    /// A jittered suggestion for how long (seconds) until the persona's next
    /// action, so a driver can schedule non-metronomic cadence.
    pub suggested_next_delay_seconds: u64,
    /// `true` only when the LLM sidecar was disabled for this tick, so no
    /// loopback network call could have happened anywhere in the pipeline
    /// (sensing, appraisal, reflection, or query generation). `false`
    /// whenever the sidecar is enabled, even if it happened not to be
    /// consulted this particular tick: a dry-run with `--llm` is NOT
    /// guaranteed network-free.
    pub no_network: bool,
}

/// A stimulus rolled and appraised on one tick, summarized for the report.
#[derive(Debug, Clone, Serialize)]
pub struct SensedStimulus {
    /// The stimulus text (what happened).
    pub text: String,
    /// The category it was about, if any.
    pub category: Option<String>,
    /// The computed appraisal salience.
    pub salience: f64,
    /// Whether it cleared the notice threshold (and so was recorded as a memory).
    pub noticed: bool,
    /// The interest seed adopted into the world-model, if any.
    pub adopted_seed: Option<String>,
}

/// The outcome of one `run-once` tick (dry-run or live).
#[derive(Debug, Clone, Serialize)]
pub struct PersonaEngineRunOutcome {
    /// The planning report for this tick.
    pub report: DryRunReport,
    /// Whether the decoy browser was actually driven (`false` for a dry-run,
    /// an idle tick, or an empty approved plan).
    pub executed: bool,
    /// How many searches were dispatched.
    pub dispatched: usize,
    /// How many planned/approved intents were skipped (rejected or not loaded).
    pub skipped: usize,
    /// The activity records produced (persisted on a live run).
    pub activity: Vec<ActivityRecord>,
}

/// Run ONE planning tick: advance the kernel, SENSE (roll an offline life
/// event, appraise it, and ingest it into the world-model if noticed), select a
/// goal, plan candidate intents (deterministic fallback unless the sidecar is
/// enabled and valid), run the Safety Gate, and truncate the approved plan to
/// the policy's per-run action budget. Mutates only `state` and drives no
/// browser or store I/O; the offline life event itself is a local computation
/// over authored `[[life_events]]`, never a real observation of the outside
/// world. It is NOT network-free when `sidecar` is an enabled LLM assistant:
/// appraisal, and on a day boundary reflection, may consult it over loopback.
/// See [`DryRunReport::no_network`].
pub fn plan_tick(
    policy: &PersonaPolicy,
    state: &mut BehaviorState,
    sidecar: &dyn SemanticAssistant,
    gate: &SafetyGate,
    now: i64,
    seed: u64,
) -> DryRunReport {
    let kernel = BehaviorKernel;
    let day_before_advance = state.world.last_reflected_day;
    let tick = kernel.advance(state, policy, now);
    let mut rng = StdRng::seed_from_u64(seed);

    // Sense: an offline life event may (rarely) fire; appraise it against the
    // persona's fixed identity and current world-model (with an optional LLM
    // salience nudge), and ingest it (record a memory, and adopt a discovered
    // interest) if it clears the notice bar. This runs BEFORE goal selection so
    // a same-tick discovery can, in principle, already nudge what gets picked
    // (its weight starts low, so in practice it takes reinforcement over
    // several sessions to really pull).
    let sensed = stimulus::roll_life_event(policy, &mut rng).map(|stim| {
        let appraisal = appraise::appraise(policy, state, &stim, sidecar);
        appraise::ingest(state, &stim, &appraisal, now);
        SensedStimulus {
            text: stim.text.clone(),
            category: stim.category.clone(),
            salience: appraisal.salience,
            noticed: appraisal.noticed,
            adopted_seed: appraisal.adopted_seed.clone(),
        }
    });

    // If the kernel's deterministic daily reflection just ran (the day
    // changed), optionally fold in a genuine LLM-written insight over the
    // recent memories. A disabled/erroring sidecar leaves this a no-op, so the
    // deterministic dominant-tag insight from `WorldState::reflect` stands
    // alone, exactly as in the no-LLM path.
    if sidecar.is_enabled() && state.world.last_reflected_day != day_before_advance {
        let recent: Vec<String> = state
            .world
            .top_memories(now, 12)
            .into_iter()
            .map(|m| m.text.clone())
            .collect();
        if let Some(insight) = sidecar.summarize_memory(&recent) {
            let insight = insight.trim();
            if !insight.is_empty() && insight.chars().count() <= 200 {
                state.world.remember(world::Memory {
                    at: now,
                    kind: world::MemoryKind::Reflection,
                    text: insight.to_string(),
                    salience: 0.6,
                    tags: vec!["llm_reflection".to_string()],
                    last_access: now,
                    hits: 0,
                });
            }
        }
    }

    let selection = GoalLayer.select(policy, state, &tick, now, &mut rng);
    // Jittered inter-arrival so a driver never schedules a metronomic cadence.
    let next_delay = kernel::next_delay_seconds(state.energy, &mut rng);

    let mut report = DryRunReport {
        persona_id: policy.id.clone(),
        policy_version: policy.schema_version,
        behavior_state: state.clone(),
        sensed,
        current_routine: tick.routine.clone(),
        idle: selection.idle,
        goal_scores: selection.scores,
        selected_goal: None,
        candidate_intents: Vec::new(),
        sidecar_used: false,
        safety_decisions: Vec::new(),
        final_plan: Vec::new(),
        suggested_next_delay_seconds: next_delay,
        no_network: !sidecar.is_enabled(),
    };

    let Some(goal) = selection.goal else {
        return report;
    };

    let intents = Planner.plan(policy, &goal, state, sidecar, seed);
    let sidecar_used = intents.iter().any(|i| i.sidecar_used);
    let (decisions, mut approved) = gate.filter(policy, &intents);
    let max = policy.action_budget.max_actions_per_run as usize;
    if approved.len() > max {
        approved.truncate(max);
    }

    report.selected_goal = Some(goal);
    report.candidate_intents = intents;
    report.sidecar_used = sidecar_used;
    report.safety_decisions = decisions;
    report.final_plan = approved;
    report
}

/// Materialize the frozen-model [`SyntheticPersona`] a policy drives for
/// execution, with a deterministic id derived from the policy id (so re-running
/// reuses the same stored persona rather than minting a new one each time).
pub fn backing_persona_for(policy: &PersonaPolicy, now: i64) -> SyntheticPersona {
    let id = deterministic_persona_id(&policy.id);
    // A nominal active window; the persona-engine re-materializes as needed and
    // does not rely on rotation here.
    let active_until = now.saturating_add(9 * 86_400_000);
    SyntheticPersona::new(
        id,
        policy.display_name.clone(),
        policy.backing_persona.age_range.clone(),
        policy.backing_persona.profession.clone(),
        policy.backing_persona.region.clone(),
        policy.backing_persona.interests.clone(),
        now,
        active_until,
    )
}

/// Build one decoy [`ActivityRecord`] from a planned intent and its outcome.
#[allow(clippy::too_many_arguments)]
pub fn make_activity_record(
    policy: &PersonaPolicy,
    routine: Option<String>,
    goal_type: &str,
    intent: &Intent,
    safety_outcome: &str,
    executor_result: &str,
    target_domain: Option<String>,
    error: Option<String>,
    now: i64,
) -> ActivityRecord {
    ActivityRecord {
        timestamp: now,
        persona_id: policy.id.clone(),
        policy_version: policy.schema_version,
        routine,
        goal: format!("{goal_type}:{}", intent.goal_id),
        action_type: intent.action_type.clone(),
        category: intent.category.clone(),
        query_seed: intent.query_seed.clone(),
        final_query: Some(intent.final_query.clone()),
        target_domain,
        dwell_seconds: None,
        egress_mode: "direct".to_string(),
        safety_outcome: safety_outcome.to_string(),
        executor_result: executor_result.to_string(),
        error,
        reason: intent.reason.clone(),
    }
}

/// Build an OFFLINE decoy-activity record: the persona did something in the real
/// world (no query, no domain, nothing on the wire).
pub fn make_offline_record(policy: &PersonaPolicy, goal: &DecoyGoal, now: i64) -> ActivityRecord {
    ActivityRecord {
        timestamp: now,
        persona_id: policy.id.clone(),
        policy_version: policy.schema_version,
        routine: Some(goal.routine.clone()),
        goal: format!("{}:{}", goal.goal_type, goal.need),
        action_type: "offline".to_string(),
        category: String::new(),
        query_seed: goal.subcategory.clone().unwrap_or_default(),
        final_query: None,
        target_domain: None,
        dwell_seconds: None,
        egress_mode: "offline".to_string(),
        safety_outcome: "allowed".to_string(),
        executor_result: "offline".to_string(),
        error: None,
        reason: goal.reason.clone(),
    }
}

/// A deterministic UUIDv4-format id derived from the policy id via FNV-1a, so the
/// materialized backing persona is stable across runs and cannot collide with a
/// randomly-minted persona by construction.
fn deterministic_persona_id(policy_id: &str) -> String {
    let a = fnv1a_64(policy_id.as_bytes(), 0xcbf2_9ce4_8422_2325);
    let b = fnv1a_64(policy_id.as_bytes(), 0x84222325_cbf29ce4 ^ 0xdead_beef);
    let mut bytes = [0u8; 16];
    bytes[..8].copy_from_slice(&a.to_be_bytes());
    bytes[8..].copy_from_slice(&b.to_be_bytes());
    // Stamp the version (4) and variant (RFC 4122) bits.
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    uuid::Uuid::from_bytes(bytes).to_string()
}

/// FNV-1a 64-bit hash with a caller-supplied offset basis.
fn fnv1a_64(data: &[u8], mut hash: u64) -> u64 {
    for &byte in data {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persona_engine::builtins;

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
    fn plan_tick_is_idle_during_quiet_hours() {
        let policy = elias();
        let mut state = BehaviorState::new("elias_rickensworth");
        let gate = SafetyGate::new();
        let report = plan_tick(&policy, &mut state, &DisabledAssistant, &gate, ts(4, 3), 1);
        assert!(report.idle);
        assert!(report.selected_goal.is_none());
        assert!(report.final_plan.is_empty());
        assert!(report.no_network);
    }

    #[test]
    fn plan_tick_produces_a_budgeted_safe_plan_in_the_evening() {
        let policy = elias();
        let gate = SafetyGate::new();
        // Search seeds until an ONLINE evening tick (a real search plan) appears;
        // offline ticks (real-world errands) are valid but carry no intents.
        for seed in 0..200u64 {
            let mut state = BehaviorState::new("elias_rickensworth");
            let report = plan_tick(
                &policy,
                &mut state,
                &DisabledAssistant,
                &gate,
                ts(4, 20),
                seed,
            );
            let online = report.selected_goal.as_ref().is_some_and(|g| g.online);
            if !report.idle && online {
                assert!(report.selected_goal.is_some());
                assert!(!report.candidate_intents.is_empty());
                // The final plan respects the per-run action budget.
                assert!(
                    report.final_plan.len() <= policy.action_budget.max_actions_per_run as usize
                );
                // Every approved intent was allowed by the gate.
                let blocklist = crate::querybank::QueryBlocklist::bundled();
                for intent in &report.final_plan {
                    assert!(!blocklist.is_blocked(&intent.final_query));
                }
                assert!(!report.sidecar_used);
                return;
            }
        }
        panic!("expected a non-idle evening plan for some seed");
    }

    #[test]
    fn backing_persona_is_deterministic_and_valid() {
        let policy = elias();
        let a = backing_persona_for(&policy, 1_700_000_000_000);
        let b = backing_persona_for(&policy, 1_700_000_000_000);
        assert_eq!(a.id, b.id);
        // A valid UUID-shaped id and no persona validation issues.
        assert_eq!(a.id.len(), 36);
        assert!(a.validate().is_empty(), "{:?}", a.validate());
        assert_eq!(a.interests, policy.backing_persona.interests);
    }
}
