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

//! The Safety Gate: mandatory, deterministic, fail-closed enforcement.
//!
//! Every planned [`Intent`] must pass the gate before it can execute. The gate
//! is deterministic and reuses the existing safety primitives rather than
//! inventing new ones:
//!
//! - the harmful-query [`QueryBlocklist`](crate::querybank::QueryBlocklist) on
//!   the final query (fails closed if the corpus does not load),
//! - the R3 auth-flow / HTTPS navigation guardrail
//!   ([`ensure_navigation_allowed`](crate::browser::isolation::ensure_navigation_allowed))
//!   on any domain hint,
//! - the policy's forbidden categories, forbidden capabilities, and allowed
//!   modules.
//!
//! A refused intent is a recorded SKIP with a reason, not a hard error: the
//! executor simply does not dispatch it. Rejections are logged locally via
//! `tracing` (no telemetry leaves the machine).

use serde::Serialize;

use crate::browser::isolation;
use crate::persona_engine::planner::Intent;
use crate::persona_engine::policy::PersonaPolicy;
use crate::querybank::QueryBlocklist;

/// The gate's decision for one intent.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SafetyDecision {
    /// The intent this decision is for.
    pub intent_id: String,
    /// Whether the intent may execute.
    pub allowed: bool,
    /// A human-readable summary of the decision.
    pub reason: String,
    /// The specific rules violated (empty when allowed).
    pub violated_rules: Vec<String>,
    /// The intent that may execute, present only when `allowed` (never a
    /// mutated/relaxed intent: the MVP gate approves as-is or rejects).
    pub sanitized_intent: Option<Intent>,
}

/// The deterministic Safety Gate. Holds a loaded harmful-query blocklist; build
/// once and reuse across a run.
#[derive(Debug)]
pub struct SafetyGate {
    blocklist: QueryBlocklist,
}

impl Default for SafetyGate {
    fn default() -> Self {
        Self::new()
    }
}

impl SafetyGate {
    /// Build a gate with the bundled harmful-query blocklist.
    pub fn new() -> Self {
        Self {
            blocklist: QueryBlocklist::bundled(),
        }
    }

    /// Whether the safety blocklist failed to load (and so every query is
    /// blocked). A health signal the caller can surface.
    pub fn blocklist_load_failed(&self) -> bool {
        self.blocklist.load_failed()
    }

    /// Evaluate one intent. Fails closed: any violated rule rejects the intent.
    pub fn evaluate(&self, policy: &PersonaPolicy, intent: &Intent) -> SafetyDecision {
        let mut violated: Vec<String> = Vec::new();

        // 1. The action's module must be allowed by the policy.
        let module = match intent.action_type.as_str() {
            "search" => "search",
            "page_visit" => "page_visit",
            other => {
                violated.push(format!("unknown_action_type:{other}"));
                ""
            }
        };
        if !module.is_empty() && !policy.is_module_allowed(module) {
            violated.push(format!("module_not_allowed:{module}"));
        }

        // 2. The action type must not itself be a forbidden capability, and no
        //    forbidden capability may be requested. (Intents only ever search or
        //    visit pages, so this is belt and suspenders.)
        if policy.is_capability_forbidden(&intent.action_type) {
            violated.push(format!("forbidden_capability:{}", intent.action_type));
        }

        // 3. Category must be allowed and not forbidden.
        if policy.forbidden_categories.contains(&intent.category) {
            violated.push(format!("forbidden_category:{}", intent.category));
        }
        if !policy.allowed_categories.contains(&intent.category) {
            violated.push(format!("category_not_allowed:{}", intent.category));
        }

        // 4. The final query must be non-empty and blocklist-clean (fail closed).
        if intent.final_query.trim().is_empty() {
            violated.push("empty_query".to_string());
        } else if self.blocklist.is_blocked(&intent.final_query) {
            violated.push("harmful_query_blocked".to_string());
        }

        // 5. Any domain hint must clear the auth-flow / HTTPS navigation guard,
        //    and must not be a policy-forbidden domain.
        for hint in &intent.allowed_domain_hints {
            let url = as_https_url(hint);
            if isolation::ensure_navigation_allowed(&url).is_err() {
                violated.push(format!("blocked_navigation:{hint}"));
            }
            if policy
                .domain_policy
                .forbidden_domains
                .iter()
                .any(|f| f == hint)
            {
                violated.push(format!("forbidden_domain:{hint}"));
            }
        }

        // 6. Budget sanity: the visit depth must not exceed the policy ceiling.
        if intent.max_depth > policy.planner_settings.max_results_to_visit {
            violated.push("exceeds_visit_budget".to_string());
        }

        if violated.is_empty() {
            SafetyDecision {
                intent_id: intent.id.clone(),
                allowed: true,
                reason: "passed all safety checks".to_string(),
                violated_rules: Vec::new(),
                sanitized_intent: Some(intent.clone()),
            }
        } else {
            // Local-only log; no telemetry leaves the machine.
            tracing::warn!(
                target: "fauxx_core::persona_engine::safety",
                intent_id = %intent.id,
                persona_id = %intent.persona_id,
                violated = ?violated,
                "safety gate rejected a planned decoy intent"
            );
            SafetyDecision {
                intent_id: intent.id.clone(),
                allowed: false,
                reason: format!("rejected: {}", violated.join(", ")),
                violated_rules: violated,
                sanitized_intent: None,
            }
        }
    }

    /// Evaluate every intent, returning the per-intent decisions plus the
    /// approved intents (in order). The approved list is what the executor runs.
    pub fn filter(
        &self,
        policy: &PersonaPolicy,
        intents: &[Intent],
    ) -> (Vec<SafetyDecision>, Vec<Intent>) {
        let mut decisions = Vec::with_capacity(intents.len());
        let mut approved = Vec::new();
        for intent in intents {
            let decision = self.evaluate(policy, intent);
            if let Some(ok) = &decision.sanitized_intent {
                approved.push(ok.clone());
            }
            decisions.push(decision);
        }
        (decisions, approved)
    }
}

/// Normalize a bare domain hint into an `https://` URL for the navigation guard.
/// A hint that already carries a scheme is passed through unchanged (so a
/// plaintext `http://` hint is correctly refused by the guard).
fn as_https_url(hint: &str) -> String {
    if hint.contains("://") {
        hint.to_string()
    } else {
        format!("https://{hint}/")
    }
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

    fn base_intent() -> Intent {
        Intent {
            id: "g#0".to_string(),
            persona_id: "elias_rickensworth".to_string(),
            routine: "weekday_evening".to_string(),
            goal_id: "g".to_string(),
            action_type: "search".to_string(),
            category: "CRAFTS".to_string(),
            query_seed: "fountain pens".to_string(),
            topic: Some("fountain pens".to_string()),
            final_query: "best blue black fountain pen ink".to_string(),
            max_depth: 1,
            dwell_range: [25, 90],
            allowed_domain_hints: Vec::new(),
            forbidden_domain_hints: Vec::new(),
            reason: "elias browses pens".to_string(),
            sidecar_used: false,
        }
    }

    #[test]
    fn a_benign_search_intent_is_allowed() {
        let gate = SafetyGate::new();
        let d = gate.evaluate(&elias(), &base_intent());
        assert!(d.allowed, "{d:?}");
        assert!(d.sanitized_intent.is_some());
        assert!(d.violated_rules.is_empty());
    }

    #[test]
    fn empty_query_is_rejected() {
        let gate = SafetyGate::new();
        let mut intent = base_intent();
        intent.final_query = "   ".to_string();
        let d = gate.evaluate(&elias(), &intent);
        assert!(!d.allowed);
        assert!(d.violated_rules.iter().any(|r| r == "empty_query"));
        assert!(d.sanitized_intent.is_none());
    }

    #[test]
    fn forbidden_category_is_rejected() {
        let gate = SafetyGate::new();
        let mut intent = base_intent();
        intent.category = "FINANCE".to_string(); // forbidden for Elias
        let d = gate.evaluate(&elias(), &intent);
        assert!(!d.allowed);
        assert!(d
            .violated_rules
            .iter()
            .any(|r| r.starts_with("forbidden_category")));
    }

    #[test]
    fn non_allowed_category_is_rejected() {
        let gate = SafetyGate::new();
        let mut intent = base_intent();
        intent.category = "GAMING".to_string(); // not in Elias's allowed set
        let d = gate.evaluate(&elias(), &intent);
        assert!(!d.allowed);
        assert!(d
            .violated_rules
            .iter()
            .any(|r| r.starts_with("category_not_allowed")));
    }

    #[test]
    fn login_action_type_is_rejected() {
        let gate = SafetyGate::new();
        let mut intent = base_intent();
        intent.action_type = "login".to_string();
        let d = gate.evaluate(&elias(), &intent);
        assert!(!d.allowed);
        // Both an unknown action type and a forbidden capability fire.
        assert!(
            d.violated_rules
                .iter()
                .any(|r| r.starts_with("forbidden_capability")
                    || r.starts_with("unknown_action_type"))
        );
    }

    #[test]
    fn auth_endpoint_domain_hint_is_rejected() {
        let gate = SafetyGate::new();
        let mut intent = base_intent();
        intent.allowed_domain_hints = vec!["accounts.google.com".to_string()];
        let d = gate.evaluate(&elias(), &intent);
        assert!(!d.allowed);
        assert!(d
            .violated_rules
            .iter()
            .any(|r| r.starts_with("blocked_navigation")));
    }

    #[test]
    fn plaintext_http_domain_hint_is_rejected() {
        let gate = SafetyGate::new();
        let mut intent = base_intent();
        intent.allowed_domain_hints = vec!["http://example.com".to_string()];
        let d = gate.evaluate(&elias(), &intent);
        assert!(!d.allowed);
        assert!(d
            .violated_rules
            .iter()
            .any(|r| r.starts_with("blocked_navigation")));
    }

    #[test]
    fn filter_partitions_allowed_and_rejected() {
        let gate = SafetyGate::new();
        let good = base_intent();
        let mut bad = base_intent();
        bad.id = "g#1".to_string();
        bad.category = "FINANCE".to_string();
        let (decisions, approved) = gate.filter(&elias(), &[good, bad]);
        assert_eq!(decisions.len(), 2);
        assert_eq!(approved.len(), 1);
        assert_eq!(approved[0].id, "g#0");
    }
}
