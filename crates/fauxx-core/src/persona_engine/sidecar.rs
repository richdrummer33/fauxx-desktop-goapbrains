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

//! The optional LLM sidecar: a NARROW, bounded semantic helper.
//!
//! The sidecar is a helper, never a driver. It may, at most, phrase an
//! already-approved query, summarize prior decoy activity into a local diary
//! line, classify page text into a coarse topic/risk label, or write a boring
//! human reason for an action the goal layer already chose. It MUST NOT choose
//! URLs, pick action types or modules, escalate volume or risk, or bypass the
//! Safety Gate.
//!
//! Rules enforced here:
//! - **Disabled by default.** The MVP ships only [`DisabledAssistant`]; there is
//!   NO cloud provider and no local model. Every method returns "unavailable" so
//!   the planner always uses its deterministic fallback.
//! - **Schema-validated output.** Any queries an assistant returns are filtered
//!   by [`validate_sidecar_queries`] before use, and the planner still runs each
//!   survivor through the harmful-query blocklist and the Safety Gate.
//! - **Deterministic fallback required.** An assistant that is disabled, errors,
//!   or returns nothing valid is indistinguishable to the planner from one that
//!   was never there.

use crate::persona_engine::policy::PersonaPolicy;

/// The bounds the planner hands the sidecar for a query-generation request. The
/// assistant must stay inside these; anything it returns is validated against
/// them by [`validate_sidecar_queries`].
#[derive(Debug, Clone)]
pub struct SidecarConstraints {
    /// The approved [`CategoryPool`](crate::persona::CategoryPool) name.
    pub category: String,
    /// The maximum number of queries to return.
    pub max_queries: usize,
    /// The maximum length (chars) of any single returned query.
    pub max_query_len: usize,
    /// Free-text topics the queries must not mention (the persona's disinterests
    /// plus any high-risk seeds).
    pub forbidden_topics: Vec<String>,
}

impl SidecarConstraints {
    /// A default constraint set for `category` allowing `max_queries` queries.
    pub fn new(category: impl Into<String>, max_queries: usize) -> Self {
        Self {
            category: category.into(),
            max_queries,
            max_query_len: 120,
            forbidden_topics: Vec::new(),
        }
    }
}

/// A narrow semantic helper. Every method is optional-return: `None` means
/// "unavailable, use the deterministic fallback". Implementations must never
/// have side effects on the executor and never see secrets.
pub trait SemanticAssistant: Send + Sync {
    /// Whether this assistant is enabled. Default `false`.
    fn is_enabled(&self) -> bool {
        false
    }

    /// Turn an approved `query_seed` into up to `constraints.max_queries`
    /// plausible, low-risk search queries within the approved category. The
    /// caller validates and blocklist-gates the result; `None` triggers the
    /// deterministic fallback.
    fn generate_search_queries(
        &self,
        _persona: &PersonaPolicy,
        _query_seed: &str,
        _constraints: &SidecarConstraints,
    ) -> Option<Vec<String>> {
        None
    }

    /// Summarize recent decoy activity into one local-only diary line.
    fn summarize_memory(&self, _recent: &[String]) -> Option<String> {
        None
    }

    /// Classify page text into coarse topic/risk labels (local-only).
    fn classify_page_text(&self, _text: &str) -> Option<Vec<String>> {
        None
    }

    /// Write a boring, human-readable reason for an already-approved action.
    fn explain_action(&self, _category: &str, _query: &str) -> Option<String> {
        None
    }

    /// Propose ONE new, closely-related topic seed near an existing interest,
    /// for the ONLY thing the deterministic path structurally cannot do:
    /// genuinely invent a new topic (rather than a new wording of one).
    /// `category` is the already-approved category the seed must belong to;
    /// `context` is a short, local description of what prompted the ask (e.g.
    /// "came across more about garden railways while searching"). The caller
    /// (`crate::persona_engine::stimulus`) treats the result as an unvetted
    /// SUGGESTION: it still must clear the category Venn and the harmful-query
    /// blocklist before it can ever be noticed, let alone adopted.
    fn propose_subseed(
        &self,
        _persona: &PersonaPolicy,
        _category: &str,
        _context: &str,
    ) -> Option<String> {
        None
    }

    /// Rate, in `[0, 1]`, how much this persona would plausibly care about a
    /// short observation. Used ONLY as a bounded MULTIPLICATIVE nudge on the
    /// deterministic appraisal salience
    /// (`crate::persona_engine::appraise::appraise`): it can amplify or damp
    /// what gets noticed, but a disabled/erroring/out-of-range response is
    /// treated as neutral (no nudge), and it can never override the hard
    /// category Venn gate that appraisal enforces before this is even called.
    fn appraise_salience(&self, _stimulus_text: &str, _category: Option<&str>) -> Option<f64> {
        None
    }
}

/// The default, disabled assistant. Every method returns "unavailable", so the
/// planner always falls back to deterministic generation. This is the ONLY
/// assistant shipped in the MVP (no cloud, no local model).
#[derive(Debug, Clone, Copy, Default)]
pub struct DisabledAssistant;

impl SemanticAssistant for DisabledAssistant {}

/// Filter sidecar-returned queries down to the ones that satisfy `constraints`:
/// non-empty, within the length limit, and mentioning none of the forbidden
/// topics. Truncates to `max_queries`. The planner still blocklist-gates every
/// survivor; this is the schema/scope check, not the safety check.
pub fn validate_sidecar_queries(
    queries: Vec<String>,
    constraints: &SidecarConstraints,
) -> Vec<String> {
    queries
        .into_iter()
        .map(|q| q.trim().to_string())
        .filter(|q| !q.is_empty() && q.chars().count() <= constraints.max_query_len)
        .filter(|q| {
            let lower = q.to_lowercase();
            !constraints
                .forbidden_topics
                .iter()
                .any(|t| !t.is_empty() && lower.contains(&t.to_lowercase()))
        })
        .take(constraints.max_queries)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persona_engine::builtins;

    #[test]
    fn disabled_assistant_returns_nothing() {
        let a = DisabledAssistant;
        assert!(!a.is_enabled());
        let policy = match builtins::get("elias_rickensworth") {
            Ok(p) => p,
            Err(e) => panic!("elias must load: {e}"),
        };
        let c = SidecarConstraints::new("CRAFTS", 5);
        assert!(a
            .generate_search_queries(&policy, "fountain pens", &c)
            .is_none());
        assert!(a.summarize_memory(&["x".to_string()]).is_none());
        assert!(a.classify_page_text("hello").is_none());
        assert!(a.explain_action("CRAFTS", "fountain pens").is_none());
        assert!(a
            .propose_subseed(&policy, "CRAFTS", "came across more about pens")
            .is_none());
        assert!(a
            .appraise_salience("something happened", Some("CRAFTS"))
            .is_none());
    }

    #[test]
    fn validation_drops_empty_overlong_and_forbidden() {
        let mut c = SidecarConstraints::new("CRAFTS", 10);
        c.max_query_len = 30;
        c.forbidden_topics = vec!["crypto".to_string()];
        let out = validate_sidecar_queries(
            vec![
                "  fountain pen ink review  ".to_string(), // trimmed, kept
                "".to_string(),                            // empty, dropped
                "   ".to_string(),                         // whitespace, dropped
                "a".repeat(50),                            // too long, dropped
                "best crypto wallet".to_string(),          // forbidden topic, dropped
            ],
            &c,
        );
        assert_eq!(out, vec!["fountain pen ink review".to_string()]);
    }

    #[test]
    fn validation_truncates_to_max_queries() {
        let c = SidecarConstraints::new("CRAFTS", 2);
        let out = validate_sidecar_queries(
            vec!["one".to_string(), "two".to_string(), "three".to_string()],
            &c,
        );
        assert_eq!(out.len(), 2);
    }
}
