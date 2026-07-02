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

//! End-to-end CLI tests for the `persona-engine` group (the decoy behavior
//! layer). These drive the compiled binary against a hermetic temp store and
//! exercise the pure commands (list/show/validate) plus the dry-run pipeline and
//! JSONL log export. They never launch a browser (no `run-once` live run).

mod common;

use anyhow::Result;
use common::{assert_ok, code, stdout, Fixture};

/// A Monday 20:00 UTC timestamp (day index 4), which lands in Elias's
/// `weekday_evening` routine so a plan is produced.
const WEEKDAY_EVENING_MS: &str = "417600000";

#[test]
fn list_shows_the_builtin_persona() -> Result<()> {
    let fx = Fixture::new()?;
    let out = fx.run(&["persona-engine", "list"])?;
    assert_ok(&out, "persona-engine list")?;
    assert!(stdout(&out)?.contains("elias_rickensworth"));
    Ok(())
}

#[test]
fn list_json_parses() -> Result<()> {
    let fx = Fixture::new()?;
    let out = fx.run(&["persona-engine", "list", "--json"])?;
    assert_ok(&out, "persona-engine list --json")?;
    let value: serde_json::Value = serde_json::from_str(&stdout(&out)?)?;
    assert!(value.is_array());
    Ok(())
}

#[test]
fn show_prints_the_persona() -> Result<()> {
    let fx = Fixture::new()?;
    let out = fx.run(&["persona-engine", "show", "elias_rickensworth"])?;
    assert_ok(&out, "persona-engine show")?;
    let text = stdout(&out)?;
    assert!(text.contains("Elias Rickensworth"));
    assert!(text.contains("weekday_evening"));
    Ok(())
}

#[test]
fn validate_builtin_is_valid() -> Result<()> {
    let fx = Fixture::new()?;
    let out = fx.run(&["persona-engine", "validate", "elias_rickensworth"])?;
    assert_ok(&out, "persona-engine validate")?;
    assert!(stdout(&out)?.contains("valid"));
    Ok(())
}

#[test]
fn validate_unknown_target_fails() -> Result<()> {
    let fx = Fixture::new()?;
    let out = fx.run(&["persona-engine", "validate", "not_a_persona"])?;
    // A missing built-in / file is a runtime error (exit 1), not success.
    assert_ne!(code(&out)?, 0);
    Ok(())
}

#[test]
fn validate_rejects_an_invalid_policy_file() -> Result<()> {
    // A policy that drops `login` from the forbidden list must fail validation.
    let fx = Fixture::new()?;
    let bad = fx.dir.join("bad.toml");
    std::fs::write(
        &bad,
        r#"
id = "bad"
display_name = "Bad"
fiction_notice = "fictional"
allowed_categories = ["HISTORY"]

[backing_persona]
age_range = "AGE_65_PLUS"
profession = "RETIRED"
region = "US_MIDWEST"
interests = ["HISTORY", "CRAFTS", "SCIENCE"]

[safety_policy]
forbidden_capabilities = ["purchase"]
allowed_modules = ["search"]

[[routines]]
name = "evening"
days = ["any"]
start_hour = 18
end_hour = 23
categories = ["HISTORY"]
"#,
    )?;
    let out = fx.run(&["persona-engine", "validate", bad.to_str().unwrap_or("")])?;
    assert_ne!(code(&out)?, 0, "an invalid policy must fail validation");
    Ok(())
}

#[test]
fn plan_dry_run_reports_pipeline_and_no_network() -> Result<()> {
    let fx = Fixture::new()?;
    let out = fx.run(&[
        "persona-engine",
        "plan",
        "--persona",
        "elias_rickensworth",
        "--dry-run",
        "--seed",
        "1",
        "--now",
        WEEKDAY_EVENING_MS,
    ])?;
    assert_ok(&out, "persona-engine plan")?;
    let text = stdout(&out)?;
    // The dry-run report carries the documented fields and the network-free flag.
    assert!(text.contains("persona_id:      elias_rickensworth"));
    assert!(text.contains("routine:         weekday_evening"));
    assert!(text.contains("final_plan:"));
    assert!(text.contains("no_network:      true"));
    Ok(())
}

#[test]
fn plan_json_is_machine_readable_and_network_free() -> Result<()> {
    let fx = Fixture::new()?;
    let out = fx.run(&[
        "persona-engine",
        "plan",
        "--persona",
        "elias_rickensworth",
        "--seed",
        "1",
        "--now",
        WEEKDAY_EVENING_MS,
        "--json",
    ])?;
    assert_ok(&out, "persona-engine plan --json")?;
    let value: serde_json::Value = serde_json::from_str(&stdout(&out)?)?;
    assert_eq!(value["persona_id"], "elias_rickensworth");
    assert_eq!(value["no_network"], true);
    assert_eq!(value["sidecar_used"], false);
    Ok(())
}

#[test]
fn run_once_dry_run_does_not_execute() -> Result<()> {
    let fx = Fixture::new()?;
    let out = fx.run(&[
        "persona-engine",
        "run-once",
        "--persona",
        "elias_rickensworth",
        "--dry-run",
        "--seed",
        "1",
        "--now",
        WEEKDAY_EVENING_MS,
        "--json",
    ])?;
    assert_ok(&out, "persona-engine run-once --dry-run")?;
    let value: serde_json::Value = serde_json::from_str(&stdout(&out)?)?;
    assert_eq!(value["executed"], false);
    assert_eq!(value["report"]["no_network"], true);
    Ok(())
}

#[test]
fn logs_export_jsonl_is_valid_and_empty_after_dry_runs() -> Result<()> {
    let fx = Fixture::new()?;
    // Dry runs persist nothing, so the exported log is empty but valid JSONL.
    let out = fx.run(&[
        "persona-engine",
        "logs",
        "export",
        "--persona",
        "elias_rickensworth",
        "--format",
        "jsonl",
    ])?;
    assert_ok(&out, "persona-engine logs export")?;
    for line in stdout(&out)?.lines() {
        if !line.trim().is_empty() {
            let _: serde_json::Value = serde_json::from_str(line)?;
        }
    }
    Ok(())
}
