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
//! a session, occasional boring pivots and skipped runs, and the daily decoy
//! budget capping activity.
//!
//! It performs NO network call and drives NO browser. Each tick runs the real
//! deterministic pipeline (kernel -> goal -> planner -> Safety Gate) via
//! `plan_tick`, and this harness plays the role of the "executor" by recording
//! the approved actions back into the state (bumping momentum, setting
//! cooldowns), exactly as a live run would after dispatching them.
//!
//! Run it with:  `cargo run -p fauxx-core --example day_in_the_life`

use std::error::Error;

use fauxx_core::persona_engine::{builtins, plan_tick};
use fauxx_core::{BehaviorState, DisabledAssistant, SafetyGate};

/// Weekday name for a UTC day index (Unix day 0 was a Thursday; Monday = 0).
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
    // Unix day 0 was a Thursday; shift so the array (Monday first) lines up.
    let dow = (day_index + 3).rem_euclid(7) as usize;
    names[dow.min(6)]
}

/// A tiny single-char energy meter.
fn energy_bar(e: f64) -> char {
    let bars = [
        '\u{2581}', '\u{2582}', '\u{2583}', '\u{2585}', '\u{2586}', '\u{2587}', '\u{2588}',
    ];
    let idx = ((e * (bars.len() as f64 - 1.0)).round() as usize).min(bars.len() - 1);
    bars[idx]
}

fn main() -> Result<(), Box<dyn Error>> {
    let policy = builtins::get("elias_rickensworth")?;
    let gate = SafetyGate::new();
    let sidecar = DisabledAssistant;
    let mut state = BehaviorState::new(&policy.id);
    let daily_cap = policy.action_budget.max_actions_per_day;

    println!(
        "A few days in the life of {} (dry simulation: no network, no browser)\n",
        policy.display_name
    );
    println!("Legend:  HH:00  <energy>  <routine>   category / topic  ->  queries\n");

    // Saturday, Sunday, Monday, Tuesday (day indices 2, 3, 4, 5).
    for day in 2..=5i64 {
        println!("---- {} ----", day_name(day));
        let mut did_something = false;
        let mut budget_noted = false;

        for hour in 6..=22u8 {
            let now = day * 86_400_000 + (hour as i64) * 3_600_000;
            // A per-tick seed so each hour makes its own draws, but the run stays
            // reproducible.
            let seed = 0xE1A5_u64 ^ ((day as u64) << 8) ^ (hour as u64);
            let report = plan_tick(&policy, &mut state, &sidecar, &gate, now, seed);
            let energy = energy_bar(report.behavior_state.energy);

            // Quiet hours (no routine): Elias potters about; keep the log sparse.
            let Some(routine) = report.current_routine.clone() else {
                continue;
            };

            // An active window but the goal layer chose to skip (anti-coherence).
            if report.idle || report.final_plan.is_empty() {
                println!("{hour:02}:00  {energy}  {routine:<16} (idle: nothing catches his eye)");
                continue;
            }

            // Respect the soft daily decoy budget (the executor's job).
            if report.behavior_state.actions_today >= daily_cap {
                if !budget_noted {
                    println!(
                        "{hour:02}:00  {energy}  {routine:<16} (daily decoy budget reached; puts the kettle on)"
                    );
                    budget_noted = true;
                }
                continue;
            }

            // "Execute": record each approved action so momentum + cooldowns evolve.
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
            let topic = goal
                .subcategory
                .as_deref()
                .unwrap_or(goal.category.as_str());
            println!(
                "{hour:02}:00  {energy}  {routine:<16} {}/{topic}  ->  {}",
                goal.category,
                queries.join(", ")
            );
            did_something = true;
        }

        if !did_something {
            println!("(a quiet day; Elias mostly watched the barometer)");
        }
        // What is Elias into at day's end?
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
        println!();
    }

    println!(
        "(Every query above passed the Safety Gate. The deterministic control layer \n\
         chose all of it; the LLM sidecar was disabled the whole time.)"
    );
    Ok(())
}
