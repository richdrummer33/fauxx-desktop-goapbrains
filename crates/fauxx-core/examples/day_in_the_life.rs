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

//! A few days in the life of Elias Rickensworth: a DRY simulation of the persona
//! engine over an hourly timeline. It threads one [`BehaviorState`] through many
//! ticks so the emergent rhythm is visible: the circadian energy curve, routines
//! turning on and off, topic momentum building, cooldowns forcing variety within
//! a session, occasional boring pivots and skipped runs, the daily decoy budget
//! capping activity, and - the point of this pass - the WORLD-MODEL: authored
//! life events he sometimes notices, occasionally becoming a brand new interest
//! that starts pulling his searches days later, all while his fixed identity
//! (core values, Big Five personality) never moves.
//!
//! It performs NO network call and drives NO browser. Each tick runs the real
//! deterministic pipeline (kernel -> sense/appraise/ingest -> goal -> planner ->
//! Safety Gate) via `plan_tick`, and this harness plays the role of the
//! "executor" by recording the approved actions back into the state (bumping
//! momentum, setting cooldowns, satisfying needs), exactly as a live run would
//! after dispatching them.
//!
//! Run it with:  `cargo run -p fauxx-core --example day_in_the_life`

// -----------------------------------------------------------------------------

// PATCHED for a live LM Studio sidecar. Diff against the original at
// crates/fauxx-core/examples/day_in_the_life.rs on branch
// claude/fauxx-persona-engine-mvp-2cchmv. Everything else is untouched;
// changes are marked "--- LLM PATCH ---".
//
// Run it (PowerShell):
//   $env:FAUXX_LLM = "1"
//   $env:FAUXX_LLM_ENDPOINT = "169.254.83.107:1234"   # your LM Studio server
//   $env:FAUXX_LLM_MODEL = "phi-4-mini-3.8b-instruct"  # exact /v1/models id - verify!
//   $env:FAUXX_SIM_DAYS = "7"                          # 7 = a week, 30 = a month
//   $env:FAUXX_LLM_API_KEY = "..."                     # only if LM Studio's
//                                                       # "Require Authentication" is on
//   cargo run -p fauxx-core --example day_in_the_life
//
// Omit FAUXX_LLM (or set it to "0") to get the original deterministic-only run.

use std::error::Error;

use fauxx_core::persona_engine::world::InterestSource;
use fauxx_core::persona_engine::{builtins, plan_tick};
// --- LLM PATCH: pull in the sidecar types instead of only DisabledAssistant ---
use fauxx_core::persona_engine::{
    LlmConfig, LmStudioAssistant, LmStudioTransport, SemanticAssistant,
};
use fauxx_core::{BehaviorState, DisabledAssistant, SafetyGate};

fn day_name(day_index: i64) -> &'static str {
    let names = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ];
    let dow = (day_index + 3).rem_euclid(7) as usize;
    names[dow.min(6)]
}

fn energy_bar(e: f64) -> char {
    let bars = [
        '\u{2581}', '\u{2582}', '\u{2583}', '\u{2585}', '\u{2586}', '\u{2587}', '\u{2588}',
    ];
    let idx = ((e * (bars.len() as f64 - 1.0)).round() as usize).min(bars.len() - 1);
    bars[idx]
}

// --- LLM PATCH: build the sidecar from env vars, boxed to a common trait
// object so the rest of main() doesn't care which concrete type it got. ---
fn build_sidecar_from_env() -> Box<dyn SemanticAssistant> {
    let enabled = std::env::var("FAUXX_LLM")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    if !enabled {
        return Box::new(DisabledAssistant);
    }
    let endpoint =
        std::env::var("FAUXX_LLM_ENDPOINT").unwrap_or_else(|_| "127.0.0.1:1234".to_string());
    let model = std::env::var("FAUXX_LLM_MODEL").unwrap_or_else(|_| "local-model".to_string());
    let timeout_ms = std::env::var("FAUXX_LLM_TIMEOUT_MS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4_000);
    let api_key = std::env::var("FAUXX_LLM_API_KEY").ok();
    eprintln!("[llm sidecar] enabled, endpoint={endpoint}, model={model}");
    let config = LlmConfig {
        enabled: true,
        endpoint,
        model,
        timeout_ms,
        api_key,
    };
    let transport = LmStudioTransport::new(&config);
    Box::new(LmStudioAssistant::new(config, transport))
}

// --- LLM PATCH: main is now async, on a multi-thread runtime, so
// LmStudioAssistant's block_in_place/block_on sync bridge has a Handle to
// find. rt-multi-thread + macros are already in the workspace's shared tokio
// feature set (root Cargo.toml), so no Cargo.toml edit is needed. ---
#[tokio::main(flavor = "multi_thread", worker_threads = 4)]
async fn main() -> Result<(), Box<dyn Error>> {
    let policy = builtins::get("elias_rickensworth")?;
    let gate = SafetyGate::new();
    let sidecar = build_sidecar_from_env(); // --- LLM PATCH (was: let sidecar = DisabledAssistant;)
    let mut state = BehaviorState::new(&policy.id);
    let daily_cap = policy.action_budget.max_actions_per_day;

    // --- LLM PATCH: configurable day range (env, default keeps the original 10-day run) ---
    let start_day: i64 = std::env::var("FAUXX_SIM_START_DAY")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2);
    let num_days: i64 = std::env::var("FAUXX_SIM_DAYS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(10);
    let end_day = start_day + num_days - 1;

    println!(
        "A few days in the life of {} (dry simulation: no network, no browser)\n",
        policy.display_name
    );
    println!(
        "Identity (fixed, never changes): {}",
        policy.identity.core_values.join(", ")
    );
    let p = &policy.identity.personality;
    println!(
        "Big Five: openness={:.2} conscientiousness={:.2} extraversion={:.2} agreeableness={:.2} neuroticism={:.2}\n",
        p.openness, p.conscientiousness, p.extraversion, p.agreeableness, p.neuroticism
    );
    println!("Legend:  HH:00  <energy>  <routine>   category / topic  ->  queries\n");

    for day in start_day..=end_day {
        println!("---- {} ----", day_name(day));
        let mut did_something = false;
        let mut budget_noted = false;

        for hour in 6..=22u8 {
            let now = day * 86_400_000 + (hour as i64) * 3_600_000;
            let seed = 0xE1A5_u64 ^ ((day as u64) << 8) ^ (hour as u64);
            // --- LLM PATCH: pass the trait object by reference ---
            let report = plan_tick(&policy, &mut state, sidecar.as_ref(), &gate, now, seed);
            let energy = energy_bar(report.behavior_state.energy);

            if let Some(sensed) = &report.sensed {
                if sensed.noticed {
                    match &sensed.adopted_seed {
                        Some(seed) => println!(
                            "{hour:02}:00  ~  (noticed) {}  -> a new interest catches him: \"{seed}\"",
                            sensed.text
                        ),
                        None => println!("{hour:02}:00  ~  (noticed) {}", sensed.text),
                    }
                    did_something = true;
                }
            }

            let Some(routine) = report.current_routine.clone() else {
                continue;
            };

            if let Some(goal) = report.selected_goal.as_ref().filter(|g| !g.online) {
                let amount = policy
                    .domain_by_need(&goal.need)
                    .map(|d| d.satisfy_amount)
                    .unwrap_or(0.4);
                state.needs.satisfy(&goal.need, amount);
                let label = goal
                    .subcategory
                    .clone()
                    .unwrap_or_else(|| goal.need.clone());
                println!("{hour:02}:00  {energy}  {routine:<16} [offline] {label}");
                did_something = true;
                continue;
            }

            if report.idle || report.final_plan.is_empty() {
                println!("{hour:02}:00  {energy}  {routine:<16} (idle: nothing catches his eye)");
                continue;
            }

            if report.behavior_state.actions_today >= daily_cap {
                if !budget_noted {
                    println!(
                        "{hour:02}:00  {energy}  {routine:<16} (daily decoy budget reached; puts the kettle on)"
                    );
                    budget_noted = true;
                }
                continue;
            }

            let goal = match &report.selected_goal {
                Some(g) => g,
                None => continue,
            };
            let queries: Vec<String> = report
                .final_plan
                .iter()
                .map(|i| format!("\"{}\"", i.final_query))
                .collect();
            for intent in &report.final_plan {
                state.record_action(&intent.category, &intent.query_seed, now);
            }
            let amount = policy
                .domain_by_need(&goal.need)
                .map(|d| d.satisfy_amount)
                .unwrap_or(0.4);
            let need = goal.need.clone();
            let topic = goal
                .subcategory
                .as_deref()
                .unwrap_or(goal.category.as_str());
            let discovered = state
                .world
                .interests
                .iter()
                .any(|i| i.seed == topic && i.source == InterestSource::Discovered);
            let marker = if discovered { " (discovered!)" } else { "" };
            println!(
                "{hour:02}:00  {energy}  {routine:<16} {}/{topic}{marker}  ->  {}",
                goal.category,
                queries.join(", ")
            );
            state.needs.satisfy(&need, amount);
            did_something = true;
        }

        if !did_something {
            println!("(a quiet day; Elias mostly watched the barometer)");
        }
        let mut scores: Vec<(&String, &f64)> = state.topic_scores.iter().collect();
        scores.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));
        let top: Vec<String> = scores
            .iter()
            .take(4)
            .map(|(k, v)| format!("{k}={v:.1}"))
            .collect();
        if !top.is_empty() {
            println!("   lingering interests: {}", top.join("  "));
        }
        let needs: Vec<String> = state
            .needs
            .levels
            .iter()
            .map(|(k, v)| format!("{k}={v:.2}"))
            .collect();
        if !needs.is_empty() {
            println!("   needs (1.0=content): {}", needs.join("  "));
        }
        println!();
    }

    let discovered: Vec<&str> = state
        .world
        .interests
        .iter()
        .filter(|i| i.source == InterestSource::Discovered)
        .map(|i| i.seed.as_str())
        .collect();
    if discovered.is_empty() {
        println!("No new interests took hold this run (try more days, or a different seed).");
    } else {
        println!(
            "New interests that took hold over the run: {}",
            discovered.join(", ")
        );
    }
    println!(
        "His core values never moved: {}\n",
        policy.identity.core_values.join(", ")
    );
    println!(
        "(Every query above passed the Safety Gate. The deterministic control layer \n\
         chose the routine, need, category and topic; the LLM sidecar (when enabled) only \n\
         phrased/appraised within that already-approved choice, and every candidate is still \n\
         blocklist- and Safety-Gate-checked before being shown. No network call, no browser.)"
    );
    Ok(())
}
