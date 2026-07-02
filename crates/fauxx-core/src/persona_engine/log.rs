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

//! The activity log: one JSONL record per planned/executed decoy action.
//!
//! Records are DECOY-ONLY synthetic data. They carry the persona's own decoy
//! decision trail (routine, goal, category, the synthetic query, the search
//! engine domain, dwell, the safety outcome). They deliberately carry NO
//! secrets, tokens, cookies, real-user identifiers, or real URLs the user
//! visited. Records persist in the encrypted store and export to a local JSONL
//! file via `persona-engine logs export`.

use serde::{Deserialize, Serialize};

/// A single decoy-activity record, serialized as one JSON object per line.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ActivityRecord {
    /// When the action was planned/executed, epoch millis.
    pub timestamp: i64,
    /// The persona policy id.
    pub persona_id: String,
    /// The policy schema version in effect.
    pub policy_version: u32,
    /// The routine active at the time, or `None` for quiet hours.
    pub routine: Option<String>,
    /// The goal (goal type + id) this action served.
    pub goal: String,
    /// The action type (MVP: `search`).
    pub action_type: String,
    /// The [`CategoryPool`](crate::persona::CategoryPool) name.
    pub category: String,
    /// The approved seed the query derived from.
    pub query_seed: String,
    /// The concrete query dispatched, when one was.
    pub final_query: Option<String>,
    /// The search engine / target domain visited, when one was.
    pub target_domain: Option<String>,
    /// The dwell time in seconds, when the action loaded.
    pub dwell_seconds: Option<u32>,
    /// The egress mode label (e.g. `direct`, `dry_run`).
    pub egress_mode: String,
    /// The Safety Gate outcome (`allowed`, or `rejected: <reasons>`).
    pub safety_outcome: String,
    /// The executor result (`dispatched`, `skipped`, `dry_run`, `error`).
    pub executor_result: String,
    /// An error message, when the action errored.
    pub error: Option<String>,
    /// A boring, human-readable reason/explanation.
    pub reason: String,
}

impl ActivityRecord {
    /// Serialize this record as a single JSONL line (no trailing newline).
    pub fn to_jsonl_line(&self) -> crate::Result<String> {
        Ok(serde_json::to_string(self)?)
    }
}

/// Render a slice of records as a JSONL document (one object per line, trailing
/// newline). This is what `persona-engine logs export --format jsonl` writes.
pub fn to_jsonl(records: &[ActivityRecord]) -> crate::Result<String> {
    let mut out = String::new();
    for record in records {
        out.push_str(&record.to_jsonl_line()?);
        out.push('\n');
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> ActivityRecord {
        ActivityRecord {
            timestamp: 1_700_000_000_000,
            persona_id: "elias_rickensworth".to_string(),
            policy_version: 1,
            routine: Some("weekday_evening".to_string()),
            goal: "continue_hobby_thread:g0".to_string(),
            action_type: "search".to_string(),
            category: "CRAFTS".to_string(),
            query_seed: "fountain pens".to_string(),
            final_query: Some("best blue black fountain pen ink".to_string()),
            target_domain: Some("duckduckgo.com".to_string()),
            dwell_seconds: Some(42),
            egress_mode: "direct".to_string(),
            safety_outcome: "allowed".to_string(),
            executor_result: "dispatched".to_string(),
            error: None,
            reason: "elias browses pens".to_string(),
        }
    }

    #[test]
    fn record_round_trips_through_jsonl() -> crate::Result<()> {
        let line = sample().to_jsonl_line()?;
        assert!(!line.contains('\n'));
        let back: ActivityRecord = serde_json::from_str(&line)?;
        assert_eq!(back, sample());
        Ok(())
    }

    #[test]
    fn jsonl_has_one_object_per_line() -> crate::Result<()> {
        let doc = to_jsonl(&[sample(), sample()])?;
        let lines: Vec<&str> = doc.lines().collect();
        assert_eq!(lines.len(), 2);
        for line in lines {
            let _: ActivityRecord = serde_json::from_str(line)?;
        }
        Ok(())
    }

    #[test]
    fn record_carries_no_secret_fields() -> crate::Result<()> {
        // The record must never grow a field that could leak secrets or real
        // user data. This freezes the key set so a future edit that adds a
        // sensitive field trips the test.
        let value = serde_json::to_value(sample())?;
        let obj = value
            .as_object()
            .ok_or_else(|| crate::CoreError::PersonaEngine("record is not an object".into()))?;
        let mut keys: Vec<&str> = obj.keys().map(String::as_str).collect();
        keys.sort_unstable();
        assert_eq!(
            keys,
            vec![
                "action_type",
                "category",
                "dwell_seconds",
                "egress_mode",
                "error",
                "executor_result",
                "final_query",
                "goal",
                "persona_id",
                "policy_version",
                "query_seed",
                "reason",
                "routine",
                "safety_outcome",
                "target_domain",
                "timestamp",
            ]
        );
        // Spot-check that no obviously-sensitive key exists.
        for banned in ["cookie", "token", "password", "secret", "key", "auth"] {
            assert!(
                !keys.iter().any(|k| k.contains(banned)),
                "banned key {banned}"
            );
        }
        Ok(())
    }
}
