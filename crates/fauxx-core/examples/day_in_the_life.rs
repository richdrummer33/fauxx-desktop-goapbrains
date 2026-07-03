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

use std::error::Error;

use fauxx_core::persona_engine::world::InterestSource;
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

    // A week and a half, so a discovered interest (which starts weak) has a
    // real chance to be reinforced and surface in a later session.
    for day in 2..=11i64 {
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

            // Sense: something may have happened, whether or not he was in an
            // active routine. Only print it when he actually NOTICED it (below
            // the notice threshold, it leaves no trace, same as real life).
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

            // Quiet hours (no routine): Elias potters about; keep the log sparse.
            let Some(routine) = report.current_routine.clone() else {
                continue;
            };

            // OFFLINE action: he attends to a need in the real world. Nothing on
            // the wire. The harness plays the executor and satisfies the need.
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

            // ONLINE action: record each approved query so momentum + cooldowns
            // evolve, and satisfy the serviced need.
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
            // Flag when the pursued topic is a DISCOVERED interest (adopted
            // from a noticed life event), not one of Elias's original seeds -
            // this is specialization actually showing up in his behavior.
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
         chose all of it; the LLM sidecar was disabled the whole time. What he noticed \n\
         and adopted came only from authored, character-consistent life events - never \n\
         from anything outside his allowed categories.)"
    );
    Ok(())
}
