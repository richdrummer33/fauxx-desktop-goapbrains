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

//! `fauxx-cli simulate <persona-id>`: preview a synthetic week of decoy
//! activity without any real browsing or network (C5 #26 P3).
//!
//! A thin shim over [`Core::simulate_week_for`]: it loads the stored persona,
//! runs the deterministic week simulator, and prints either the raw
//! [`SimulatedWeek`] as JSON (for scripted consumers) or a human-readable
//! day-by-day timeline plus a category-frequency breakdown. The same
//! `(persona, intensity, seed)` always yields an identical result; pass a
//! different `--seed` to re-roll the week.

use fauxx_core::{Config, Core};

use crate::cli::SimulateArgs;

/// Preview a synthetic week for the stored persona named in `args`.
pub async fn run(config: Config, args: SimulateArgs) -> anyhow::Result<()> {
    let core = Core::open(config).await?;
    let week = core
        .simulate_week_for(&args.persona_id, args.intensity.into(), args.seed)
        .await?;

    if args.json {
        println!("{}", serde_json::to_string_pretty(&week)?);
        return Ok(());
    }

    println!(
        "simulate week: persona={}  intensity={:?}  seed={}  total-queries={}",
        week.persona_id,
        week.intensity,
        week.seed,
        week.total_queries()
    );
    for session in &week.sessions {
        println!(
            "  day {:02}  start={}  {} quer{}",
            session.day,
            format_time_of_day(session.start_secs),
            session.queries.len(),
            if session.queries.len() == 1 { "y" } else { "ies" },
        );
    }
    println!("category breakdown (highest first):");
    for (category, count) in week.category_counts() {
        println!("  {category:<30}  {count}");
    }
    Ok(())
}

/// Format a second-of-day offset as `HH:MM:SS` (local wall-clock label only;
/// the simulator works in second-offsets from midnight, not absolute time).
fn format_time_of_day(secs_into_day: i64) -> String {
    let s = secs_into_day.rem_euclid(86_400);
    let (h, m, sec) = (s / 3_600, (s % 3_600) / 60, s % 60);
    format!("{h:02}:{m:02}:{sec:02}")
}
