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

//! Needs / motives: the Sims-style decaying drives that give a persona a life
//! beyond one hobby.
//!
//! Each named need holds a satisfaction level in `[0, 1]` (`1.0` = content,
//! `0.0` = starving for it). Time DEPLETES needs; actions SATISFY them. The goal
//! layer services whichever need is most deficient (weighted, not greedy), so a
//! persona naturally alternates between indulging a hobby, doing upkeep, and
//! attending to wellbeing, instead of only ever browsing.
//!
//! Needs are declared by the persona's `domains` (a domain both serves a need
//! and knows how to satisfy it, online or offline). Pure and deterministic given
//! the elapsed time.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// The neutral starting level for a need never seen before.
const DEFAULT_LEVEL: f64 = 0.5;

/// A persona's need/motive levels, keyed by need name. Persisted in the
/// behavior state so a persona's drives carry across runs.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct NeedState {
    /// Need name -> satisfaction level in `[0, 1]`.
    pub levels: BTreeMap<String, f64>,
}

impl NeedState {
    /// The satisfaction level for `need` (`DEFAULT_LEVEL` when unseen).
    pub fn level(&self, need: &str) -> f64 {
        self.levels.get(need).copied().unwrap_or(DEFAULT_LEVEL)
    }

    /// The deficit for `need` in `[0, 1]` (how badly it wants attention).
    pub fn deficit(&self, need: &str) -> f64 {
        (1.0 - self.level(need)).clamp(0.0, 1.0)
    }

    /// Ensure every `(name, _)` need exists, seeded at the neutral level.
    pub fn ensure(&mut self, names: impl IntoIterator<Item = String>) {
        for name in names {
            self.levels.entry(name).or_insert(DEFAULT_LEVEL);
        }
    }

    /// Deplete a need by `rate_per_hour * elapsed_hours` (clamped to `[0, 1]`).
    pub fn deplete(&mut self, need: &str, rate_per_hour: f64, elapsed_hours: f64) {
        if elapsed_hours <= 0.0 {
            return;
        }
        let level = self.levels.entry(need.to_string()).or_insert(DEFAULT_LEVEL);
        *level = (*level - rate_per_hour * elapsed_hours).clamp(0.0, 1.0);
    }

    /// Replenish a need by `amount` (clamped to `[0, 1]`).
    pub fn satisfy(&mut self, need: &str, amount: f64) {
        let level = self.levels.entry(need.to_string()).or_insert(DEFAULT_LEVEL);
        *level = (*level + amount).clamp(0.0, 1.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unseen_need_is_neutral() {
        let n = NeedState::default();
        assert_eq!(n.level("hobby"), DEFAULT_LEVEL);
        assert!((n.deficit("hobby") - 0.5).abs() < 1e-9);
    }

    #[test]
    fn depletes_over_time_and_satisfies() {
        let mut n = NeedState::default();
        n.ensure(["hobby".to_string()]);
        n.deplete("hobby", 0.1, 3.0); // 0.5 - 0.3 = 0.2
        assert!((n.level("hobby") - 0.2).abs() < 1e-9);
        assert!(n.deficit("hobby") > 0.5);
        n.satisfy("hobby", 0.5); // 0.2 + 0.5 = 0.7
        assert!((n.level("hobby") - 0.7).abs() < 1e-9);
    }

    #[test]
    fn levels_clamp_to_unit_interval() {
        let mut n = NeedState::default();
        n.satisfy("x", 10.0);
        assert_eq!(n.level("x"), 1.0);
        n.deplete("x", 100.0, 1.0);
        assert_eq!(n.level("x"), 0.0);
    }

    #[test]
    fn zero_elapsed_is_a_noop() {
        let mut n = NeedState::default();
        n.ensure(["hobby".to_string()]);
        n.deplete("hobby", 0.5, 0.0);
        assert_eq!(n.level("hobby"), DEFAULT_LEVEL);
    }
}
