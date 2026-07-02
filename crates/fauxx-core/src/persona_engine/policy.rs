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

//! Persona policy: who a decoy persona is ALLOWED to be.
//!
//! A [`PersonaPolicy`] is the top of the persona-engine pipeline. It is a
//! human-authored TOML document that declares the persona's identity, the
//! topics it may and may not touch, its routines, its activity/risk budget, and
//! its hard safety restrictions. It is deliberately SEPARATE from the frozen
//! cross-device [`SyntheticPersona`](crate::persona::SyntheticPersona) wire
//! model: the policy is the desktop-local behavioral BRAIN, and it references a
//! [`BackingPersona`] (mapped onto the frozen 32 [`CategoryPool`] values) so the
//! existing query generator and decoy browser can execute for it without the
//! wire contract changing.
//!
//! The policy chooses; the LLM never does. Every list here (allowed categories,
//! forbidden capabilities, allowed modules, routines, budgets) is enforced
//! downstream by the goal layer and the deterministic Safety Gate.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::persona::{AgeRange, CategoryPool, Profession, Region};

/// The current persona-policy schema version. Bumped when the policy shape
/// changes incompatibly; [`PersonaPolicy::validate`] rejects a newer version it
/// cannot understand rather than silently misreading a policy.
pub const POLICY_SCHEMA_VERSION: u32 = 1;

/// A persona policy: the full, validated description of a decoy persona's
/// allowed behavior. Parsed from TOML via [`PersonaPolicy::from_toml_str`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PersonaPolicy {
    /// Policy schema version (see [`POLICY_SCHEMA_VERSION`]). Defaults to the
    /// current version for policies that omit it.
    #[serde(default = "default_schema_version")]
    pub schema_version: u32,
    /// Stable, human-readable policy id (e.g. `elias_rickensworth`). Used on the
    /// CLI and as the key for the persona's persisted behavior state.
    pub id: String,
    /// Display name shown to the operator.
    pub display_name: String,
    /// A required, explicit notice that this persona is FICTIONAL and harmless
    /// and does not impersonate a real person. Non-empty is enforced by
    /// [`validate`](Self::validate).
    pub fiction_notice: String,
    /// Coarse region / locale constraint for the persona.
    #[serde(default)]
    pub region_mode: RegionMode,
    /// Egress / isolation requirements the executor must satisfy (fail closed).
    #[serde(default)]
    pub egress_requirements: EgressRequirements,
    /// The frozen-model persona this policy materializes for execution.
    pub backing_persona: BackingPersona,
    /// [`CategoryPool`] names the goal layer may choose from.
    #[serde(default)]
    pub allowed_categories: Vec<String>,
    /// [`CategoryPool`] names the persona must NEVER touch (belt and suspenders
    /// over the global harmful-query blocklist).
    #[serde(default)]
    pub forbidden_categories: Vec<String>,
    /// Free-text topic seeds per category (the persona's flavor, e.g. "fountain
    /// pens"), keyed by [`CategoryPool`] name. Used to nudge query generation, for
    /// logging, and (later) the LLM sidecar. Never a URL and never executed
    /// directly; every derived query is still Safety-Gated.
    #[serde(default)]
    pub topic_seeds: BTreeMap<String, Vec<String>>,
    /// Free-text topics the persona actively avoids (flavor + a soft down-weight).
    #[serde(default)]
    pub disinterests: Vec<String>,
    /// The persona's daily routines (time-of-day windows and their goals).
    #[serde(default)]
    pub routines: Vec<Routine>,
    /// Topic-decay / anti-coherence tuning.
    #[serde(default)]
    pub topic_decay: TopicDecay,
    /// Bounded activity budget per run and per day.
    #[serde(default)]
    pub action_budget: ActionBudget,
    /// Domain allowlist / blocklist policy for page visits.
    #[serde(default)]
    pub domain_policy: DomainPolicy,
    /// Hard safety restrictions (forbidden capabilities, allowed modules).
    #[serde(default)]
    pub safety_policy: SafetyPolicy,
    /// Planner tuning (candidate count, dwell ranges, sidecar toggle).
    #[serde(default)]
    pub planner_settings: PlannerSettings,
    /// Life domains: the needs/motives this persona services and how (online
    /// search vs offline real-world errand). Additive: when empty, a single
    /// online `hobby` domain is synthesized from [`allowed_categories`], so a
    /// policy that predates domains behaves exactly as before.
    #[serde(default)]
    pub domains: Vec<Domain>,
}

/// A life domain: a need/motive the persona services, and how it does so. A
/// domain with categories and a high `online_bias` is mostly satisfied by decoy
/// SEARCHES; a domain with no categories (or a low `online_bias`) is satisfied
/// OFFLINE, in the real world, emitting nothing on the wire. Offline domains are
/// how a persona attends to sensitive real-life needs (health, errands) WITHOUT
/// ever turning them into decoy query signal.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Domain {
    /// The need this domain serves (e.g. `hobby`, `upkeep`, `wellbeing`).
    pub name: String,
    /// [`CategoryPool`] names this domain searches online (empty = offline-only).
    #[serde(default)]
    pub categories: Vec<String>,
    /// Probability in `[0, 1]` that servicing this need is an ONLINE search
    /// rather than an offline errand. `0.0` = always offline.
    #[serde(default = "default_online_bias")]
    pub online_bias: f64,
    /// How fast this need depletes, per hour.
    #[serde(default = "default_decay_per_hour")]
    pub decay_per_hour: f64,
    /// How much one action replenishes this need.
    #[serde(default = "default_satisfy_amount")]
    pub satisfy_amount: f64,
    /// Flavor label for an offline action (e.g. `goes for a walk`).
    #[serde(default)]
    pub offline_label: Option<String>,
    /// Goal-type labels this domain favors.
    #[serde(default)]
    pub goal_types: Vec<String>,
}

fn default_online_bias() -> f64 {
    0.9
}
fn default_decay_per_hour() -> f64 {
    0.06
}
fn default_satisfy_amount() -> f64 {
    0.4
}

fn default_schema_version() -> u32 {
    POLICY_SCHEMA_VERSION
}

/// Coarse region / locale constraint. Free-form `mode` label plus an optional
/// desktop-local home-location hint (never synced to the phone).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RegionMode {
    /// A label such as `us_local`; interpreted by the operator, not enforced
    /// against a geo database in the MVP.
    #[serde(default)]
    pub mode: String,
    /// Optional freeform home-location label for flavor.
    #[serde(default)]
    pub home_location: Option<String>,
}

impl Default for RegionMode {
    fn default() -> Self {
        Self {
            mode: "us_local".to_string(),
            home_location: None,
        }
    }
}

/// Egress / isolation requirements. All default to the safest setting so a
/// policy that omits the block still runs fail-closed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EgressRequirements {
    /// The decoy must run from an isolated profile verifiably separate from any
    /// real browser profile (always enforced by the browser layer regardless).
    #[serde(default = "default_true")]
    pub require_isolated_profile: bool,
    /// Decoy navigation must be HTTPS (enforced by the isolation guard too).
    #[serde(default = "default_true")]
    pub require_https: bool,
}

impl Default for EgressRequirements {
    fn default() -> Self {
        Self {
            require_isolated_profile: true,
            require_https: true,
        }
    }
}

fn default_true() -> bool {
    true
}

/// The frozen-model persona this policy materializes into the store for
/// execution. All fields are [`CategoryPool`]/[`AgeRange`]/[`Profession`]/
/// [`Region`] enum NAMES (validated by [`PersonaPolicy::validate`]).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BackingPersona {
    /// [`AgeRange`] enum name (e.g. `AGE_65_PLUS`).
    pub age_range: String,
    /// [`Profession`] enum name (e.g. `RETIRED`).
    pub profession: String,
    /// [`Region`] enum name (e.g. `US_MIDWEST`).
    pub region: String,
    /// [`CategoryPool`] enum names; a well-formed persona carries 3..=5.
    pub interests: Vec<String>,
}

/// Topic-decay and anti-coherence tuning. "Too coherent" is a fingerprint, so
/// the persona occasionally pivots, skips a day, and lets topics cool off.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TopicDecay {
    /// Hours for a topic's momentum score to halve.
    #[serde(default = "default_half_life")]
    pub half_life_hours: f64,
    /// Chance the goal layer takes a boring, off-momentum pivot instead of the
    /// hottest topic.
    #[serde(default = "default_pivot_probability")]
    pub pivot_probability: f64,
    /// Chance a whole run is skipped (an idle day).
    #[serde(default = "default_skip_day_probability")]
    pub skip_day_probability: f64,
}

impl Default for TopicDecay {
    fn default() -> Self {
        Self {
            half_life_hours: default_half_life(),
            pivot_probability: default_pivot_probability(),
            skip_day_probability: default_skip_day_probability(),
        }
    }
}

fn default_half_life() -> f64 {
    36.0
}
fn default_pivot_probability() -> f64 {
    0.15
}
fn default_skip_day_probability() -> f64 {
    0.10
}

/// Bounded activity budget. The goal layer and Safety Gate never exceed these.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ActionBudget {
    /// Max decoy actions (searches / page visits) in one run.
    #[serde(default = "default_max_actions_per_run")]
    pub max_actions_per_run: u32,
    /// Max wall-clock minutes one run may span.
    #[serde(default = "default_max_minutes_per_run")]
    pub max_minutes_per_run: u32,
    /// Soft daily ceiling on decoy actions (advisory in the MVP).
    #[serde(default = "default_max_actions_per_day")]
    pub max_actions_per_day: u32,
}

impl Default for ActionBudget {
    fn default() -> Self {
        Self {
            max_actions_per_run: default_max_actions_per_run(),
            max_minutes_per_run: default_max_minutes_per_run(),
            max_actions_per_day: default_max_actions_per_day(),
        }
    }
}

fn default_max_actions_per_run() -> u32 {
    3
}
fn default_max_minutes_per_run() -> u32 {
    20
}
fn default_max_actions_per_day() -> u32 {
    12
}

/// Domain allow/deny policy for page visits. The MVP prefers curated allowlists
/// and search over arbitrary open-web navigation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DomainPolicy {
    /// Prefer the curated `allow_domains` over open-web navigation.
    #[serde(default = "default_true")]
    pub prefer_allowlist: bool,
    /// Curated allowlisted domains (optional in the MVP; search is the default
    /// module).
    #[serde(default)]
    pub allow_domains: Vec<String>,
    /// Explicitly forbidden domains (in addition to the global auth blocklist).
    #[serde(default)]
    pub forbidden_domains: Vec<String>,
}

impl Default for DomainPolicy {
    fn default() -> Self {
        Self {
            prefer_allowlist: true,
            allow_domains: Vec::new(),
            forbidden_domains: Vec::new(),
        }
    }
}

/// Hard safety restrictions. `forbidden_capabilities` names things the persona
/// must never do; `allowed_modules` names the only executor modules it may use.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SafetyPolicy {
    /// Capabilities the persona must never exercise (login, submit_form, ...).
    #[serde(default = "default_forbidden_capabilities")]
    pub forbidden_capabilities: Vec<String>,
    /// The only executor modules the persona may use (MVP: search, page_visit).
    #[serde(default = "default_allowed_modules")]
    pub allowed_modules: Vec<String>,
}

impl Default for SafetyPolicy {
    fn default() -> Self {
        Self {
            forbidden_capabilities: default_forbidden_capabilities(),
            allowed_modules: default_allowed_modules(),
        }
    }
}

/// The MVP hard-forbidden capability list. The Safety Gate treats these as
/// always-refused regardless of what a policy declares; a policy may only ADD to
/// them, never remove one.
pub const HARD_FORBIDDEN_CAPABILITIES: &[&str] = &[
    "login",
    "create_account",
    "submit_form",
    "comment",
    "post",
    "upload",
    "message",
    "review",
    "purchase",
    "checkout",
    "book",
    "apply",
    "vote",
    "contact",
    "click_ads",
    "bypass_captcha",
];

fn default_forbidden_capabilities() -> Vec<String> {
    HARD_FORBIDDEN_CAPABILITIES
        .iter()
        .map(|s| s.to_string())
        .collect()
}

fn default_allowed_modules() -> Vec<String> {
    vec!["search".to_string(), "page_visit".to_string()]
}

/// Planner tuning: how many candidate intents to build, the dwell range, and
/// whether the (MVP-disabled) LLM sidecar may be consulted.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlannerSettings {
    /// Whether the planner may consult the LLM sidecar. Even when `true`, the MVP
    /// ships only the disabled assistant, so the deterministic fallback is used.
    #[serde(default)]
    pub use_llm_sidecar: bool,
    /// How many candidate intents the planner produces per goal.
    #[serde(default = "default_max_candidate_intents")]
    pub max_candidate_intents: u32,
    /// Minimum dwell seconds on a visited result.
    #[serde(default = "default_dwell_min")]
    pub dwell_seconds_min: u32,
    /// Maximum dwell seconds on a visited result.
    #[serde(default = "default_dwell_max")]
    pub dwell_seconds_max: u32,
    /// Max results the persona will visit per search (MVP keeps this at 1).
    #[serde(default = "default_max_results_to_visit")]
    pub max_results_to_visit: u32,
}

impl Default for PlannerSettings {
    fn default() -> Self {
        Self {
            use_llm_sidecar: false,
            max_candidate_intents: default_max_candidate_intents(),
            dwell_seconds_min: default_dwell_min(),
            dwell_seconds_max: default_dwell_max(),
            max_results_to_visit: default_max_results_to_visit(),
        }
    }
}

fn default_max_candidate_intents() -> u32 {
    5
}
fn default_dwell_min() -> u32 {
    25
}
fn default_dwell_max() -> u32 {
    90
}
fn default_max_results_to_visit() -> u32 {
    1
}

/// A daily routine: a time-of-day window (on weekdays, weekends, or both) and
/// the goals / categories it favors.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Routine {
    /// Routine name (e.g. `weekday_evening`).
    pub name: String,
    /// Which days this routine applies to: `weekday`, `weekend`, or `any`.
    #[serde(default = "default_days")]
    pub days: Vec<String>,
    /// Window start hour, local 0..=23 (inclusive).
    pub start_hour: u8,
    /// Window end hour, local 0..=23 (exclusive upper bound on the hour).
    pub end_hour: u8,
    /// Goal-type labels this routine favors (e.g. `continue_hobby_thread`).
    #[serde(default)]
    pub goal_types: Vec<String>,
    /// [`CategoryPool`] names this routine favors.
    #[serde(default)]
    pub categories: Vec<String>,
}

fn default_days() -> Vec<String> {
    vec!["any".to_string()]
}

impl Routine {
    /// Whether this routine is active on the given day/hour. `is_weekend`
    /// selects the day class; `hour` is local 0..=23.
    pub fn covers(&self, is_weekend: bool, hour: u8) -> bool {
        let day_ok = self.days.iter().any(|d| match d.as_str() {
            "any" => true,
            "weekend" => is_weekend,
            "weekday" => !is_weekend,
            _ => false,
        });
        day_ok && hour >= self.start_hour && hour < self.end_hour
    }
}

/// A single problem found by [`PersonaPolicy::validate`]. Surfaced to the CLI
/// `persona-engine validate` command; a policy with any issue is not usable.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum PolicyIssue {
    /// The schema version is newer than this build understands.
    UnsupportedSchemaVersion(u32),
    /// A required field is empty (carries the field name).
    EmptyField(&'static str),
    /// A category name is not a known [`CategoryPool`] (carries where + the name).
    UnknownCategory { field: &'static str, name: String },
    /// A category appears in both allowed and forbidden lists.
    AllowedAndForbidden(String),
    /// The backing persona's age-range name is unknown.
    UnknownAgeRange(String),
    /// The backing persona's profession name is unknown.
    UnknownProfession(String),
    /// The backing persona's region name is unknown.
    UnknownRegion(String),
    /// The backing persona's interest count is outside 3..=5.
    BackingInterestCount(usize),
    /// A routine has an invalid hour window (start >= end, or hour > 23).
    InvalidRoutineWindow(String),
    /// A budget field is zero (a persona that can do nothing).
    ZeroBudget(&'static str),
    /// The dwell range is inverted (min > max).
    InvertedDwellRange,
    /// A forbidden capability required by the hard list is missing from the
    /// policy's `forbidden_capabilities`.
    MissingHardForbidden(String),
    /// A domain references a category that is not in `allowed_categories`.
    DomainCategoryNotAllowed { domain: String, name: String },
    /// A domain's `online_bias` is outside `[0, 1]`.
    InvalidOnlineBias { domain: String },
}

impl PersonaPolicy {
    /// Parse a policy from a TOML document. A syntax or shape error is a
    /// [`CoreError::PersonaEngine`](crate::error::CoreError::PersonaEngine); this
    /// does NOT run [`validate`](Self::validate) (call it separately so the CLI
    /// can report every issue at once).
    pub fn from_toml_str(toml_str: &str) -> crate::Result<Self> {
        toml::from_str(toml_str).map_err(|e| {
            crate::CoreError::PersonaEngine(format!("failed to parse persona policy TOML: {e}"))
        })
    }

    /// Validate the policy against the known enums and structural rules. Returns
    /// every problem found (empty when the policy is valid).
    pub fn validate(&self) -> Vec<PolicyIssue> {
        let mut issues = Vec::new();

        if self.schema_version > POLICY_SCHEMA_VERSION {
            issues.push(PolicyIssue::UnsupportedSchemaVersion(self.schema_version));
        }
        if self.id.trim().is_empty() {
            issues.push(PolicyIssue::EmptyField("id"));
        }
        if self.display_name.trim().is_empty() {
            issues.push(PolicyIssue::EmptyField("display_name"));
        }
        if self.fiction_notice.trim().is_empty() {
            issues.push(PolicyIssue::EmptyField("fiction_notice"));
        }

        // Category name checks across every place a category name may appear.
        check_categories("allowed_categories", &self.allowed_categories, &mut issues);
        check_categories(
            "forbidden_categories",
            &self.forbidden_categories,
            &mut issues,
        );
        for key in self.topic_seeds.keys() {
            if CategoryPool::from_name(key).is_none() {
                issues.push(PolicyIssue::UnknownCategory {
                    field: "topic_seeds",
                    name: key.clone(),
                });
            }
        }
        for routine in &self.routines {
            check_categories("routines.categories", &routine.categories, &mut issues);
        }

        // Allowed and forbidden must not overlap.
        for cat in &self.allowed_categories {
            if self.forbidden_categories.contains(cat) {
                issues.push(PolicyIssue::AllowedAndForbidden(cat.clone()));
            }
        }

        // Backing persona enums + interest count.
        if AgeRange::from_name(&self.backing_persona.age_range).is_none() {
            issues.push(PolicyIssue::UnknownAgeRange(
                self.backing_persona.age_range.clone(),
            ));
        }
        if Profession::from_name(&self.backing_persona.profession).is_none() {
            issues.push(PolicyIssue::UnknownProfession(
                self.backing_persona.profession.clone(),
            ));
        }
        if Region::from_name(&self.backing_persona.region).is_none() {
            issues.push(PolicyIssue::UnknownRegion(
                self.backing_persona.region.clone(),
            ));
        }
        check_categories(
            "backing_persona.interests",
            &self.backing_persona.interests,
            &mut issues,
        );
        if !crate::persona::INTEREST_COUNT.contains(&self.backing_persona.interests.len()) {
            issues.push(PolicyIssue::BackingInterestCount(
                self.backing_persona.interests.len(),
            ));
        }

        // Routines: non-empty and each window valid.
        if self.routines.is_empty() {
            issues.push(PolicyIssue::EmptyField("routines"));
        }
        for routine in &self.routines {
            if routine.start_hour >= routine.end_hour
                || routine.start_hour > 23
                || routine.end_hour > 24
            {
                issues.push(PolicyIssue::InvalidRoutineWindow(routine.name.clone()));
            }
        }

        // Budgets must let the persona do something.
        if self.action_budget.max_actions_per_run == 0 {
            issues.push(PolicyIssue::ZeroBudget("max_actions_per_run"));
        }
        if self.action_budget.max_minutes_per_run == 0 {
            issues.push(PolicyIssue::ZeroBudget("max_minutes_per_run"));
        }
        if self.planner_settings.dwell_seconds_min > self.planner_settings.dwell_seconds_max {
            issues.push(PolicyIssue::InvertedDwellRange);
        }

        // Domains: each searched category must be a known, ALLOWED category, and
        // the online bias must be a probability. Offline-only domains (no
        // categories) are fine.
        for domain in &self.domains {
            for cat in &domain.categories {
                if CategoryPool::from_name(cat).is_none() {
                    issues.push(PolicyIssue::UnknownCategory {
                        field: "domains.categories",
                        name: cat.clone(),
                    });
                } else if !self.allowed_categories.contains(cat) {
                    issues.push(PolicyIssue::DomainCategoryNotAllowed {
                        domain: domain.name.clone(),
                        name: cat.clone(),
                    });
                }
            }
            if !(0.0..=1.0).contains(&domain.online_bias) {
                issues.push(PolicyIssue::InvalidOnlineBias {
                    domain: domain.name.clone(),
                });
            }
        }

        // Every hard-forbidden capability must be present (a policy may add, not
        // remove). Belt and suspenders: the Safety Gate refuses them anyway.
        for cap in HARD_FORBIDDEN_CAPABILITIES {
            if !self
                .safety_policy
                .forbidden_capabilities
                .iter()
                .any(|c| c == cap)
            {
                issues.push(PolicyIssue::MissingHardForbidden((*cap).to_string()));
            }
        }

        issues
    }

    /// The allowed categories as parsed [`CategoryPool`] values (unknown names
    /// dropped; [`validate`](Self::validate) flags those separately).
    pub fn allowed_category_pool(&self) -> Vec<CategoryPool> {
        self.allowed_categories
            .iter()
            .filter_map(|n| CategoryPool::from_name(n))
            .collect()
    }

    /// The persona's life domains, synthesizing a single online `hobby` domain
    /// from [`allowed_categories`](Self::allowed_categories) when none are
    /// declared (so a pre-domains policy behaves exactly as before).
    pub fn effective_domains(&self) -> Vec<Domain> {
        if !self.domains.is_empty() {
            return self.domains.clone();
        }
        vec![Domain {
            name: "hobby".to_string(),
            categories: self.allowed_categories.clone(),
            online_bias: 1.0,
            decay_per_hour: default_decay_per_hour(),
            satisfy_amount: default_satisfy_amount(),
            offline_label: None,
            goal_types: Vec::new(),
        }]
    }

    /// The domain serving `need`, if any.
    pub fn domain_by_need(&self, need: &str) -> Option<Domain> {
        self.effective_domains()
            .into_iter()
            .find(|d| d.name == need)
    }

    /// The first routine covering `is_weekend`/`hour`, or `None` (quiet hours).
    pub fn routine_for(&self, is_weekend: bool, hour: u8) -> Option<&Routine> {
        self.routines.iter().find(|r| r.covers(is_weekend, hour))
    }

    /// Topic seeds for a category (empty slice when none are declared).
    pub fn topic_seeds_for(&self, category: CategoryPool) -> &[String] {
        self.topic_seeds
            .get(category.as_name())
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    /// Whether a capability is forbidden by this policy OR the hard list.
    pub fn is_capability_forbidden(&self, capability: &str) -> bool {
        HARD_FORBIDDEN_CAPABILITIES.contains(&capability)
            || self
                .safety_policy
                .forbidden_capabilities
                .iter()
                .any(|c| c == capability)
    }

    /// Whether an executor module is allowed by this policy.
    pub fn is_module_allowed(&self, module: &str) -> bool {
        self.safety_policy
            .allowed_modules
            .iter()
            .any(|m| m == module)
    }
}

/// Push an [`UnknownCategory`](PolicyIssue::UnknownCategory) for every name that
/// is not a known [`CategoryPool`].
fn check_categories(field: &'static str, names: &[String], issues: &mut Vec<PolicyIssue>) {
    for name in names {
        if CategoryPool::from_name(name).is_none() {
            issues.push(PolicyIssue::UnknownCategory {
                field,
                name: name.clone(),
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn minimal_toml() -> &'static str {
        r#"
id = "test_persona"
display_name = "Test Persona"
fiction_notice = "A fictional, harmless decoy persona."
allowed_categories = ["TECHNOLOGY", "SCIENCE"]

[backing_persona]
age_range = "AGE_35_44"
profession = "ENGINEER"
region = "US_WEST"
interests = ["TECHNOLOGY", "SCIENCE", "HISTORY"]

[[routines]]
name = "evening"
days = ["any"]
start_hour = 18
end_hour = 23
categories = ["TECHNOLOGY"]
"#
    }

    #[test]
    fn parses_minimal_policy_and_applies_defaults() -> crate::Result<()> {
        let policy = PersonaPolicy::from_toml_str(minimal_toml())?;
        assert_eq!(policy.id, "test_persona");
        // schema_version defaults to the current version when omitted.
        assert_eq!(policy.schema_version, POLICY_SCHEMA_VERSION);
        // Budgets + safety defaults are populated.
        assert_eq!(policy.action_budget.max_actions_per_run, 3);
        assert!(policy
            .safety_policy
            .forbidden_capabilities
            .iter()
            .any(|c| c == "login"));
        assert_eq!(
            policy.safety_policy.allowed_modules,
            vec!["search".to_string(), "page_visit".to_string()]
        );
        Ok(())
    }

    #[test]
    fn minimal_policy_is_valid() -> crate::Result<()> {
        let policy = PersonaPolicy::from_toml_str(minimal_toml())?;
        assert!(policy.validate().is_empty(), "{:?}", policy.validate());
        Ok(())
    }

    #[test]
    fn unknown_category_is_flagged() -> crate::Result<()> {
        let toml =
            minimal_toml().replace("\"TECHNOLOGY\", \"SCIENCE\"", "\"TECHNOLOGY\", \"NOPE\"");
        let policy = PersonaPolicy::from_toml_str(&toml)?;
        assert!(policy.validate().iter().any(|i| matches!(
            i,
            PolicyIssue::UnknownCategory { name, .. } if name == "NOPE"
        )));
        Ok(())
    }

    #[test]
    fn allowed_and_forbidden_overlap_is_flagged() -> crate::Result<()> {
        let mut policy = PersonaPolicy::from_toml_str(minimal_toml())?;
        policy.forbidden_categories = vec!["TECHNOLOGY".to_string()];
        assert!(policy
            .validate()
            .iter()
            .any(|i| matches!(i, PolicyIssue::AllowedAndForbidden(c) if c == "TECHNOLOGY")));
        Ok(())
    }

    #[test]
    fn newer_schema_version_is_rejected() -> crate::Result<()> {
        let mut policy = PersonaPolicy::from_toml_str(minimal_toml())?;
        policy.schema_version = POLICY_SCHEMA_VERSION + 1;
        assert!(policy
            .validate()
            .iter()
            .any(|i| matches!(i, PolicyIssue::UnsupportedSchemaVersion(_))));
        Ok(())
    }

    #[test]
    fn empty_fiction_notice_is_flagged() -> crate::Result<()> {
        let mut policy = PersonaPolicy::from_toml_str(minimal_toml())?;
        policy.fiction_notice = "   ".to_string();
        assert!(policy
            .validate()
            .iter()
            .any(|i| matches!(i, PolicyIssue::EmptyField("fiction_notice"))));
        Ok(())
    }

    #[test]
    fn invalid_routine_window_is_flagged() -> crate::Result<()> {
        let mut policy = PersonaPolicy::from_toml_str(minimal_toml())?;
        policy.routines[0].start_hour = 22;
        policy.routines[0].end_hour = 6;
        assert!(policy
            .validate()
            .iter()
            .any(|i| matches!(i, PolicyIssue::InvalidRoutineWindow(_))));
        Ok(())
    }

    #[test]
    fn routine_covers_day_and_hour() -> crate::Result<()> {
        let policy = PersonaPolicy::from_toml_str(minimal_toml())?;
        let r = &policy.routines[0];
        assert!(r.covers(false, 20));
        assert!(r.covers(true, 18));
        assert!(!r.covers(false, 17));
        assert!(!r.covers(false, 23));
        Ok(())
    }

    #[test]
    fn capability_and_module_helpers() -> crate::Result<()> {
        let policy = PersonaPolicy::from_toml_str(minimal_toml())?;
        assert!(policy.is_capability_forbidden("login"));
        assert!(policy.is_capability_forbidden("purchase"));
        assert!(!policy.is_capability_forbidden("search"));
        assert!(policy.is_module_allowed("search"));
        assert!(!policy.is_module_allowed("form_fill"));
        Ok(())
    }

    #[test]
    fn no_domains_synthesizes_a_hobby_domain() -> crate::Result<()> {
        let policy = PersonaPolicy::from_toml_str(minimal_toml())?;
        let domains = policy.effective_domains();
        assert_eq!(domains.len(), 1);
        assert_eq!(domains[0].name, "hobby");
        assert_eq!(domains[0].categories, policy.allowed_categories);
        Ok(())
    }

    #[test]
    fn domain_category_must_be_allowed_and_known() -> crate::Result<()> {
        let toml = format!(
            "{}\n[[domains]]\nname = \"junk\"\ncategories = [\"FINANCE\", \"NOPE\"]\n",
            minimal_toml()
        );
        let policy = PersonaPolicy::from_toml_str(&toml)?;
        let issues = policy.validate();
        // FINANCE is a real category but not in allowed_categories; NOPE is unknown.
        assert!(issues.iter().any(
            |i| matches!(i, PolicyIssue::DomainCategoryNotAllowed { name, .. } if name == "FINANCE")
        ));
        assert!(issues
            .iter()
            .any(|i| matches!(i, PolicyIssue::UnknownCategory { name, .. } if name == "NOPE")));
        Ok(())
    }

    #[test]
    fn invalid_online_bias_is_flagged() -> crate::Result<()> {
        let toml = format!(
            "{}\n[[domains]]\nname = \"hobby\"\ncategories = [\"TECHNOLOGY\"]\nonline_bias = 1.5\n",
            minimal_toml()
        );
        let policy = PersonaPolicy::from_toml_str(&toml)?;
        assert!(policy
            .validate()
            .iter()
            .any(|i| matches!(i, PolicyIssue::InvalidOnlineBias { domain } if domain == "hobby")));
        Ok(())
    }

    #[test]
    fn missing_hard_forbidden_capability_is_flagged() -> crate::Result<()> {
        let mut policy = PersonaPolicy::from_toml_str(minimal_toml())?;
        // A policy that tries to drop `login` from the forbidden list is flagged.
        policy.safety_policy.forbidden_capabilities = vec!["purchase".to_string()];
        assert!(policy
            .validate()
            .iter()
            .any(|i| matches!(i, PolicyIssue::MissingHardForbidden(c) if c == "login")));
        Ok(())
    }
}
