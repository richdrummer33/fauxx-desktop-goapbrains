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

//! `fauxx-cli persona-engine ...`: the decoy behavior layer (the Sims-like
//! persona engine).
//!
//! A thin shim over the core's persona-engine API. `list`/`show`/`validate` are
//! pure and never open the store; `plan`/`run-once`/`logs` open the core so the
//! persona's behavior state and activity log persist. `plan` (and `run-once
//! --dry-run`) never drive the browser and never write to the store; they are
//! only network-free when `--llm` is not passed (see `report.no_network`).

use std::path::{Path, PathBuf};

use anyhow::{bail, Context};
use fauxx_core::persona_engine::builtins;
use fauxx_core::{
    Config, Core, DryRunReport, LlmConfig, PersonaEngineRunOutcome, PersonaPolicy, PolicySummary,
};

use crate::cli::{PersonaEngineCommand, PersonaEngineLogFormat, PersonaEngineLogsCommand};

/// Dispatch a `persona-engine` subcommand.
pub async fn run(config: Config, command: PersonaEngineCommand) -> anyhow::Result<()> {
    match command {
        PersonaEngineCommand::List { json } => list(json),
        PersonaEngineCommand::Show { name, json } => show(&name, json),
        PersonaEngineCommand::Validate { target } => validate(&target),
        PersonaEngineCommand::Plan {
            persona,
            dry_run: _,
            seed,
            now,
            json,
            llm,
            llm_endpoint,
            llm_model,
            llm_api_key,
        } => {
            let llm_config = llm_config_from_flags(llm, llm_endpoint, llm_model, llm_api_key);
            plan(config, &persona, seed, now, json, llm_config).await
        }
        PersonaEngineCommand::RunOnce {
            persona,
            dry_run,
            decoy_id,
            seed,
            now,
            json,
            llm,
            llm_endpoint,
            llm_model,
            llm_api_key,
        } => {
            let llm_config = llm_config_from_flags(llm, llm_endpoint, llm_model, llm_api_key);
            run_once(
                config, &persona, dry_run, decoy_id, seed, now, json, llm_config,
            )
            .await
        }
        PersonaEngineCommand::Logs { command } => match command {
            PersonaEngineLogsCommand::Export {
                persona,
                format,
                out,
            } => export_logs(config, &persona, format, out).await,
        },
    }
}

/// List the built-in policies (pure; no store).
fn list(json: bool) -> anyhow::Result<()> {
    let summaries: Vec<PolicySummary> = builtins::list()
        .iter()
        .filter_map(|id| {
            builtins::get(id)
                .ok()
                .map(|p| PolicySummary::from_policy(&p, true))
        })
        .collect();
    if json {
        println!("{}", serde_json::to_string_pretty(&summaries)?);
        return Ok(());
    }
    if summaries.is_empty() {
        println!("no built-in personas");
        return Ok(());
    }
    for s in &summaries {
        println!(
            "{}  \"{}\"  routines={}  categories={}",
            s.id,
            s.display_name,
            s.routine_count,
            s.allowed_categories.join(",")
        );
    }
    Ok(())
}

/// Show one policy (pure; no store).
fn show(name: &str, json: bool) -> anyhow::Result<()> {
    let policy = resolve_policy(name)?;
    if json {
        println!("{}", serde_json::to_string_pretty(&policy)?);
        return Ok(());
    }
    println!("id:            {}", policy.id);
    println!("display_name:  {}", policy.display_name);
    println!("schema:        v{}", policy.schema_version);
    println!("fiction:       {}", policy.fiction_notice.trim());
    if !policy.identity.core_values.is_empty() {
        println!("core_values:   {}", policy.identity.core_values.join(", "));
    }
    let p = &policy.identity.personality;
    println!(
        "personality:   O={:.2} C={:.2} E={:.2} A={:.2} N={:.2}",
        p.openness, p.conscientiousness, p.extraversion, p.agreeableness, p.neuroticism
    );
    println!("allowed:       {}", policy.allowed_categories.join(", "));
    println!("forbidden:     {}", policy.forbidden_categories.join(", "));
    if !policy.life_events.is_empty() {
        println!("life_events:   {} authored", policy.life_events.len());
    }
    println!(
        "modules:       {}",
        policy.safety_policy.allowed_modules.join(", ")
    );
    println!(
        "budget:        {} actions / {} min per run",
        policy.action_budget.max_actions_per_run, policy.action_budget.max_minutes_per_run
    );
    println!("routines:");
    for r in &policy.routines {
        println!(
            "  - {} [{}] {:02}:00-{:02}:00  categories={}",
            r.name,
            r.days.join("/"),
            r.start_hour,
            r.end_hour,
            r.categories.join(",")
        );
    }
    Ok(())
}

/// Validate a policy (built-in id or .toml path). Non-zero issues fail the
/// command with a runtime error listing each problem.
fn validate(target: &str) -> anyhow::Result<()> {
    let policy = resolve_policy(target)?;
    let issues = policy.validate();
    if issues.is_empty() {
        println!("{}: valid", policy.id);
        return Ok(());
    }
    eprintln!("{}: {} issue(s):", policy.id, issues.len());
    for issue in &issues {
        eprintln!("  - {issue:?}");
    }
    bail!(
        "policy {} is invalid ({} issue(s))",
        policy.id,
        issues.len()
    );
}

/// Dry-run the pipeline (opens the core to read persisted state; never writes).
#[allow(clippy::too_many_arguments)]
async fn plan(
    config: Config,
    persona: &str,
    seed: u64,
    now: Option<i64>,
    json: bool,
    llm: Option<LlmConfig>,
) -> anyhow::Result<()> {
    let policy = resolve_policy(persona)?;
    let now = now.unwrap_or_else(now_millis);
    let core = Core::open(config).await?;
    let report = core.persona_engine_plan(&policy, now, seed, llm).await?;
    if json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        print_report(&report);
    }
    Ok(())
}

/// Run one tick (dry-run or live).
#[allow(clippy::too_many_arguments)]
async fn run_once(
    config: Config,
    persona: &str,
    dry_run: bool,
    decoy_id: Option<String>,
    seed: u64,
    now: Option<i64>,
    json: bool,
    llm: Option<LlmConfig>,
) -> anyhow::Result<()> {
    let policy = resolve_policy(persona)?;
    let now = now.unwrap_or_else(now_millis);
    let core = Core::open(config).await?;
    let outcome = core
        .persona_engine_run_once(&policy, decoy_id, now, seed, dry_run, llm)
        .await?;
    if json {
        println!("{}", serde_json::to_string_pretty(&outcome)?);
    } else {
        print_run_outcome(&outcome);
    }
    Ok(())
}

/// Build an [`LlmConfig`] from the CLI's `--llm`/`--llm-endpoint`/`--llm-model`/
/// `--llm-api-key` flags, or `None` when `--llm` was not passed (the
/// deterministic-only path).
fn llm_config_from_flags(
    enabled: bool,
    endpoint: String,
    model: String,
    api_key: Option<String>,
) -> Option<LlmConfig> {
    if !enabled {
        return None;
    }
    Some(LlmConfig {
        enabled: true,
        endpoint,
        model,
        api_key,
        ..LlmConfig::default()
    })
}

/// Export the decoy activity log as JSONL (to a file or stdout).
async fn export_logs(
    config: Config,
    persona: &str,
    format: PersonaEngineLogFormat,
    out: Option<PathBuf>,
) -> anyhow::Result<()> {
    let PersonaEngineLogFormat::Jsonl = format;
    let core = Core::open(config).await?;
    let jsonl = core.persona_engine_export_logs_jsonl(persona).await?;
    match out {
        Some(path) => {
            std::fs::write(&path, jsonl.as_bytes())
                .with_context(|| format!("writing activity log to {}", path.display()))?;
            println!(
                "wrote {} activity record(s) to {}",
                jsonl.lines().count(),
                path.display()
            );
        }
        None => print!("{jsonl}"),
    }
    Ok(())
}

/// Resolve `target` to a policy: a built-in id, else a path to a .toml file.
fn resolve_policy(target: &str) -> anyhow::Result<PersonaPolicy> {
    if builtins::is_builtin(target) {
        return builtins::get(target).map_err(anyhow::Error::from);
    }
    let path = Path::new(target);
    if path.exists() {
        let toml = std::fs::read_to_string(path)
            .with_context(|| format!("reading policy file {}", path.display()))?;
        return PersonaPolicy::from_toml_str(&toml).map_err(anyhow::Error::from);
    }
    bail!(
        "no built-in persona and no policy file named {:?} (built-ins: {})",
        target,
        builtins::list().join(", ")
    )
}

/// Print a human-readable dry-run report.
fn print_report(report: &DryRunReport) {
    println!("persona_id:      {}", report.persona_id);
    println!("policy_version:  v{}", report.policy_version);
    println!(
        "routine:         {}",
        report.current_routine.as_deref().unwrap_or("(quiet hours)")
    );
    let s = &report.behavior_state;
    println!(
        "state:           energy={:.2} curiosity={:.2} boredom={:.2}",
        s.energy, s.curiosity, s.boredom
    );
    match &report.sensed {
        Some(sensed) if sensed.noticed => {
            let adopted = sensed
                .adopted_seed
                .as_deref()
                .map(|s| format!("  -> new interest: \"{s}\""))
                .unwrap_or_default();
            println!(
                "sensed:          (noticed, salience={:.2}) {}{adopted}",
                sensed.salience, sensed.text
            );
        }
        Some(sensed) => println!(
            "sensed:          (unnoticed, salience={:.2}) {}",
            sensed.salience, sensed.text
        ),
        None => {}
    }
    if !s.topic_scores.is_empty() {
        let scores: Vec<String> = s
            .topic_scores
            .iter()
            .map(|(k, v)| format!("{k}={v:.2}"))
            .collect();
        println!("topic_scores:    {}", scores.join(" "));
    }
    if !s.needs.levels.is_empty() {
        let needs: Vec<String> = s
            .needs
            .levels
            .iter()
            .map(|(k, v)| format!("{k}={v:.2}"))
            .collect();
        println!("needs:           {}", needs.join(" "));
    }
    if !s.world.interests.is_empty() {
        let discovered = s
            .world
            .interests
            .iter()
            .filter(|i| i.source == fauxx_core::persona_engine::world::InterestSource::Discovered)
            .count();
        println!(
            "interests:       {} total ({} discovered)  memories={}",
            s.world.interests.len(),
            discovered,
            s.world.memories.len()
        );
    }
    if report.idle {
        println!("decision:        IDLE (no action this tick)");
    }
    if !report.goal_scores.is_empty() {
        let scores: Vec<String> = report
            .goal_scores
            .iter()
            .map(|(k, v)| format!("{k}={v:.2}"))
            .collect();
        println!("goal_scores:     {}", scores.join(" "));
    }
    if let Some(goal) = &report.selected_goal {
        println!("selected_goal:   {} ({})", goal.goal_type, goal.id);
        println!(
            "need:            {}  [{}]",
            goal.need,
            if goal.online {
                "online search"
            } else {
                "offline errand"
            }
        );
        println!("action_type:     {}", goal.action_type);
        if goal.online {
            println!(
                "category:        {}{}",
                goal.category,
                goal.subcategory
                    .as_deref()
                    .map(|s| format!(" / {s}"))
                    .unwrap_or_default()
            );
        }
        println!("reason:          {}", goal.reason);
    }
    println!("sidecar_used:    {}", report.sidecar_used);
    if !report.candidate_intents.is_empty() {
        println!("candidate_intents:");
        for i in &report.candidate_intents {
            println!("  - [{}] {}", i.category, i.final_query);
        }
    }
    if !report.safety_decisions.is_empty() {
        println!("safety:");
        for d in &report.safety_decisions {
            let verdict = if d.allowed { "ALLOW" } else { "REJECT" };
            println!("  - {} {}: {}", verdict, d.intent_id, d.reason);
        }
    }
    println!("final_plan:");
    if let Some(goal) = report.selected_goal.as_ref().filter(|g| !g.online) {
        println!(
            "  (offline errand: {})",
            goal.subcategory.as_deref().unwrap_or(&goal.need)
        );
    } else if report.final_plan.is_empty() {
        println!("  (nothing to do)");
    } else {
        for i in &report.final_plan {
            println!(
                "  - [{}] \"{}\"  dwell={}-{}s  visit<={}",
                i.category, i.final_query, i.dwell_range[0], i.dwell_range[1], i.max_depth
            );
        }
    }
    println!(
        "next_action_in:  ~{} min (jittered cadence)",
        report.suggested_next_delay_seconds / 60
    );
    println!("no_network:      {}", report.no_network);
}

/// Print a human-readable run outcome.
fn print_run_outcome(outcome: &PersonaEngineRunOutcome) {
    print_report(&outcome.report);
    println!(
        "executed:        {}  (dispatched={}, skipped={})",
        outcome.executed, outcome.dispatched, outcome.skipped
    );
}

/// Wall-clock time in epoch millis (0 if the clock predates the epoch).
fn now_millis() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}
