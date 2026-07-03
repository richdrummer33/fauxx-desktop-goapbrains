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

//! The world-model: a bounded memory stream and a living interest graph.
//!
//! This is "modeled awareness with meaning": what the persona has NOTICED
//! (memories) and what it now CARES ABOUT (interests), as distinct from the raw
//! behavioral counters in [`crate::persona_engine::kernel::BehaviorState`]. Two
//! deliberately simple, deterministic structures:
//!
//! - [`Memory`]: a timestamped, salience-weighted observation. Retrieval and
//!   forgetting use a cheap ACT-R-style activation (exponential recency decay
//!   plus a frequency bonus), not literal ACT-R base-level learning; the goal is
//!   "what still matters now", not psychological fidelity.
//! - [`Interest`]: a node in a small interest graph. AUTHORED nodes (from the
//!   policy's `topic_seeds`) are permanent gravity wells that never fall below
//!   their `base_weight`. DISCOVERED nodes (adopted from a noticed stimulus) are
//!   light and decay if not reinforced, and can only ever be adopted near an
//!   authored node's category (the Venn/gravity constraint lives in
//!   `appraise.rs`, which is the only writer of discovered interests).
//!
//! Everything here is pure and deterministic given `now` and an injected RNG;
//! there is no clock read and no I/O.

use serde::{Deserialize, Serialize};

use crate::persona_engine::policy::PersonaPolicy;

/// Max memories retained; beyond this the lowest-activation memory is forgotten.
const MEMORY_CAP: usize = 120;
/// Memory activation half-life, in hours (how fast a memory "fades").
const MEMORY_HALF_LIFE_HOURS: f64 = 72.0;
/// Bonus added to activation per repeat access (a re-noticed thing sticks).
const MEMORY_HIT_BONUS: f64 = 0.15;

/// Discovered-interest weight half-life, in hours (how fast an un-reinforced
/// discovery fades back out, distinct from the authored gravity wells).
const DISCOVERED_HALF_LIFE_HOURS: f64 = 96.0;
/// Below this weight a discovered interest is pruned from the graph entirely.
const PRUNE_FLOOR: f64 = 0.05;
/// Weight bump a hit (the interest was pursued) applies, on top of decay.
const REINFORCE_BUMP: f64 = 0.25;

/// What kind of observation a [`Memory`] records.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryKind {
    /// A real-world happening (from an authored `[[life_events]]` entry).
    OfflineEvent,
    /// Something noticed from an online search/page.
    OnlineDiscovery,
    /// A higher-level insight synthesized by periodic reflection.
    Reflection,
}

/// One noticed, meaningful observation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Memory {
    /// When this was noticed, epoch millis.
    pub at: i64,
    /// What kind of observation this is.
    pub kind: MemoryKind,
    /// The observation text (a short, human-readable line).
    pub text: String,
    /// The salience it was noticed with (see `appraise.rs`), in `[0, 1]`-ish.
    pub salience: f64,
    /// Free-text tags (e.g. category names, seed names) for retrieval/reflection.
    pub tags: Vec<String>,
    /// The last time this memory was retrieved/reinforced, epoch millis.
    pub last_access: i64,
    /// How many times this memory has been retrieved/reinforced.
    pub hits: u32,
}

impl Memory {
    /// This memory's current activation (retrieval strength) at `now`: salience
    /// decayed by elapsed time since `last_access`, plus a small per-hit bonus.
    /// Higher is more "alive" in mind right now.
    pub fn activation(&self, now: i64) -> f64 {
        let elapsed_hours = ((now - self.last_access).max(0)) as f64 / 3_600_000.0;
        let decay = 0.5_f64.powf(elapsed_hours / MEMORY_HALF_LIFE_HOURS);
        self.salience * decay + MEMORY_HIT_BONUS * (self.hits as f64)
    }
}

/// Where an [`Interest`] node came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InterestSource {
    /// Declared in the policy's `topic_seeds`; a permanent gravity well.
    Authored,
    /// Adopted at runtime from a noticed, appraised stimulus.
    Discovered,
}

/// One node in the interest graph: a topic seed the persona may pursue.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Interest {
    /// The topic seed text (e.g. `"blotting paper"`).
    pub seed: String,
    /// The [`CategoryPool`](crate::persona::CategoryPool) name it belongs to.
    pub category: String,
    /// The floor weight never decayed below (0 for a discovered node).
    pub base_weight: f64,
    /// The current weight; how strongly this pulls seed selection.
    pub weight: f64,
    /// Authored (permanent) or discovered (can fade and be pruned).
    pub source: InterestSource,
    /// When this node entered the graph, epoch millis.
    pub discovered_at: i64,
    /// How many times this seed has been pursued.
    pub hits: u32,
    /// The last time this seed was pursued, epoch millis (0 if never).
    pub last_used: i64,
}

impl Interest {
    /// The weight to use for selection RIGHT NOW: authored nodes never drop
    /// below their gravity floor; discovered nodes use their (already-decayed)
    /// weight as-is.
    pub fn effective_weight(&self) -> f64 {
        match self.source {
            InterestSource::Authored => self.weight.max(self.base_weight),
            InterestSource::Discovered => self.weight,
        }
    }
}

/// The persona's world-model: what it has noticed and what it now cares about.
/// Persisted as part of [`BehaviorState`](crate::persona_engine::BehaviorState).
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct WorldState {
    /// The bounded memory stream, oldest first.
    pub memories: Vec<Memory>,
    /// The interest graph (authored + discovered nodes).
    pub interests: Vec<Interest>,
    /// The UTC day index [`reflect`](Self::reflect) last ran for (0 = never).
    pub last_reflected_day: i64,
}

impl WorldState {
    /// Seed the interest graph from the policy's authored `topic_seeds`, if the
    /// graph is empty (first use, or a policy predating the world-model). A
    /// no-op once seeded, so re-running never resets discovered interests or
    /// authored-weight tuning done at runtime.
    pub fn ensure_seeded(&mut self, policy: &PersonaPolicy, now: i64) {
        if !self.interests.is_empty() {
            return;
        }
        for (category, seeds) in &policy.topic_seeds {
            for seed in seeds {
                self.interests.push(Interest {
                    seed: seed.clone(),
                    category: category.clone(),
                    base_weight: 1.0,
                    weight: 1.0,
                    source: InterestSource::Authored,
                    discovered_at: now,
                    hits: 0,
                    last_used: 0,
                });
            }
        }
    }

    /// The interest nodes for `category`.
    pub fn interests_for_category<'a>(&'a self, category: &str) -> Vec<&'a Interest> {
        self.interests
            .iter()
            .filter(|i| i.category == category)
            .collect()
    }

    /// Record that `seed` was pursued at `now`: reinforce its weight and bump
    /// its hit/last-used bookkeeping. A no-op if `seed` is not in the graph
    /// (callers should adopt/seed first).
    pub fn reinforce(&mut self, seed: &str, now: i64) {
        if let Some(interest) = self.interests.iter_mut().find(|i| i.seed == seed) {
            interest.weight = (interest.weight + REINFORCE_BUMP).min(4.0);
            interest.hits = interest.hits.saturating_add(1);
            interest.last_used = now;
        }
    }

    /// Adopt a new DISCOVERED interest, or reinforce it if already present.
    /// Callers (appraisal) are responsible for the Venn/blocklist gate BEFORE
    /// calling this; this method only manages the graph itself.
    pub fn adopt(&mut self, seed: &str, category: &str, initial_weight: f64, now: i64) {
        if let Some(existing) = self.interests.iter_mut().find(|i| i.seed == seed) {
            existing.weight = (existing.weight + initial_weight).min(4.0);
            existing.hits = existing.hits.saturating_add(1);
            return;
        }
        self.interests.push(Interest {
            seed: seed.to_string(),
            category: category.to_string(),
            base_weight: 0.0,
            weight: initial_weight.max(0.01),
            source: InterestSource::Discovered,
            discovered_at: now,
            hits: 1,
            last_used: 0,
        });
    }

    /// Push a memory, then forget the lowest-activation memory beyond the cap.
    pub fn remember(&mut self, memory: Memory) {
        self.memories.push(memory);
        if self.memories.len() > MEMORY_CAP {
            let now = self.memories.last().map(|m| m.at).unwrap_or(0);
            let weakest = self
                .memories
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    a.activation(now)
                        .partial_cmp(&b.activation(now))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i);
            if let Some(i) = weakest {
                self.memories.remove(i);
            }
        }
    }

    /// The `top_n` memories by activation at `now` (for reflection/reporting).
    pub fn top_memories(&self, now: i64, top_n: usize) -> Vec<&Memory> {
        let mut scored: Vec<(&Memory, f64)> = self
            .memories
            .iter()
            .map(|m| (m, m.activation(now)))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.into_iter().take(top_n).map(|(m, _)| m).collect()
    }

    /// Daily reflection: decay discovered interests toward zero (authored ones
    /// are untouched, protected by their gravity floor), prune the ones that
    /// fell below [`PRUNE_FLOOR`], and, if there is a dominant recent tag among
    /// noticed memories, synthesize one higher-level [`Memory`] insight. Callers
    /// gate this to once per UTC day via `last_reflected_day`.
    pub fn reflect(&mut self, now: i64, elapsed_hours: f64) {
        if elapsed_hours > 0.0 {
            let factor = 0.5_f64.powf(elapsed_hours / DISCOVERED_HALF_LIFE_HOURS);
            for interest in &mut self.interests {
                if interest.source == InterestSource::Discovered {
                    interest.weight *= factor;
                }
            }
        }
        self.interests
            .retain(|i| i.source == InterestSource::Authored || i.weight >= PRUNE_FLOOR);

        if let Some(insight) = self.dominant_tag_insight(now) {
            self.remember(Memory {
                at: now,
                kind: MemoryKind::Reflection,
                text: insight.0,
                salience: 0.5,
                tags: vec![insight.1],
                last_access: now,
                hits: 0,
            });
        }
    }

    /// If one tag dominates the recent noticed memories (>= 3 occurrences among
    /// the most-active ones), return a synthesized insight line and that tag.
    /// The deterministic stand-in for LLM-quality reflection.
    fn dominant_tag_insight(&self, now: i64) -> Option<(String, String)> {
        use std::collections::HashMap;
        let recent = self.top_memories(now, 12);
        let mut counts: HashMap<&str, usize> = HashMap::new();
        for m in &recent {
            for tag in &m.tags {
                *counts.entry(tag.as_str()).or_insert(0) += 1;
            }
        }
        let (tag, count) = counts.into_iter().max_by_key(|(_, c)| *c)?;
        if count < 3 {
            return None;
        }
        Some((format!("keeps returning to {tag} lately"), tag.to_string()))
    }
}

/// Pick a seed proportional to its `effective_weight`, deterministically given
/// `rng`. `None` if `candidates` is empty or every weight is non-positive.
pub fn weighted_pick<'a>(
    candidates: &[&'a Interest],
    rng: &mut impl rand::RngExt,
) -> Option<&'a Interest> {
    let total: f64 = candidates
        .iter()
        .map(|i| i.effective_weight().max(0.0))
        .sum();
    if candidates.is_empty() || total <= 0.0 {
        return None;
    }
    let mut pick = rng.random::<f64>() * total;
    for &c in candidates {
        pick -= c.effective_weight().max(0.0);
        if pick <= 0.0 {
            return Some(c);
        }
    }
    candidates.last().copied()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn mem(at: i64, salience: f64, tags: &[&str]) -> Memory {
        Memory {
            at,
            kind: MemoryKind::OnlineDiscovery,
            text: "x".to_string(),
            salience,
            tags: tags.iter().map(|s| s.to_string()).collect(),
            last_access: at,
            hits: 0,
        }
    }

    #[test]
    fn memory_activation_decays_with_time() {
        let m = mem(0, 1.0, &[]);
        let fresh = m.activation(0);
        let stale = m.activation((MEMORY_HALF_LIFE_HOURS * 3_600_000.0) as i64);
        assert!(
            (stale - fresh / 2.0).abs() < 1e-6,
            "expected ~half at one half-life"
        );
    }

    #[test]
    fn memory_stream_forgets_the_weakest_beyond_cap() {
        let mut w = WorldState::default();
        for i in 0..MEMORY_CAP + 5 {
            w.remember(mem(i as i64 * 1000, 0.1, &[]));
        }
        assert_eq!(w.memories.len(), MEMORY_CAP);
    }

    #[test]
    fn ensure_seeded_is_idempotent_and_only_fires_once() {
        use crate::persona_engine::builtins;
        let policy = match builtins::get("elias_rickensworth") {
            Ok(p) => p,
            Err(e) => panic!("elias must load: {e}"),
        };
        let mut w = WorldState::default();
        w.ensure_seeded(&policy, 0);
        let count = w.interests.len();
        assert!(count > 0);
        // Adopt a discovered interest, then re-seed: must not reset the graph.
        w.adopt("nib grinding", "CRAFTS", 0.3, 1);
        w.ensure_seeded(&policy, 2);
        assert_eq!(w.interests.len(), count + 1);
    }

    #[test]
    fn authored_interest_never_drops_below_gravity_floor() {
        let mut w = WorldState::default();
        w.interests.push(Interest {
            seed: "fountain pens".to_string(),
            category: "CRAFTS".to_string(),
            base_weight: 1.0,
            weight: 1.0,
            source: InterestSource::Authored,
            discovered_at: 0,
            hits: 0,
            last_used: 0,
        });
        // Even a long reflect-decay never sinks an authored node below its floor.
        w.reflect(1_000_000_000, 10_000.0);
        let i = &w.interests[0];
        assert!((i.effective_weight() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn discovered_interest_decays_and_gets_pruned() {
        let mut w = WorldState::default();
        w.adopt("model train weathering", "OUTDOOR_RECREATION", 0.2, 0);
        assert_eq!(w.interests.len(), 1);
        // Many half-lives with no reinforcement: it decays away and is pruned.
        w.reflect(1, DISCOVERED_HALF_LIFE_HOURS * 20.0);
        assert!(w.interests.is_empty());
    }

    #[test]
    fn adopt_reinforces_an_existing_node_instead_of_duplicating() {
        let mut w = WorldState::default();
        w.adopt("model train weathering", "OUTDOOR_RECREATION", 0.2, 0);
        w.adopt("model train weathering", "OUTDOOR_RECREATION", 0.2, 100);
        assert_eq!(w.interests.len(), 1);
        assert!(w.interests[0].weight > 0.2);
        assert_eq!(w.interests[0].hits, 2);
    }

    #[test]
    fn weighted_pick_favors_higher_weight_over_many_draws() {
        let low = Interest {
            seed: "low".to_string(),
            category: "CRAFTS".to_string(),
            base_weight: 0.0,
            weight: 0.1,
            source: InterestSource::Discovered,
            discovered_at: 0,
            hits: 0,
            last_used: 0,
        };
        let high = Interest {
            seed: "high".to_string(),
            category: "CRAFTS".to_string(),
            base_weight: 0.0,
            weight: 2.0,
            source: InterestSource::Discovered,
            discovered_at: 0,
            hits: 0,
            last_used: 0,
        };
        let candidates = vec![&low, &high];
        let mut high_wins = 0;
        for s in 0..200u64 {
            let mut rng = StdRng::seed_from_u64(s);
            if weighted_pick(&candidates, &mut rng).map(|i| i.seed.as_str()) == Some("high") {
                high_wins += 1;
            }
        }
        assert!(
            high_wins > 150,
            "expected the heavier interest to dominate, got {high_wins}/200"
        );
    }

    #[test]
    fn dominant_tag_reflection_is_synthesized() {
        let mut w = WorldState::default();
        for i in 0..4 {
            w.remember(mem(i * 1000, 0.5, &["CRAFTS"]));
        }
        w.reflect(10_000, 1.0);
        assert!(w
            .memories
            .iter()
            .any(|m| m.kind == MemoryKind::Reflection && m.tags.contains(&"CRAFTS".to_string())));
    }
}
