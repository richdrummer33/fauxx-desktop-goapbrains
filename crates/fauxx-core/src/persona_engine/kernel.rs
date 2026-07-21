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

//! The behavior kernel: a small Sims-like state model.
//!
//! [`BehaviorState`] tracks the persona's evolving mood and history (energy,
//! curiosity, boredom, per-topic momentum, recent actions, cooldowns). The
//! kernel advances it deterministically from the wall clock: it applies a
//! circadian energy curve, decays topic momentum with a half-life, prunes
//! expired cooldowns, and resolves the current routine (or quiet hours).
//!
//! Everything here is pure and deterministic given `now` (epoch millis), so
//! tests drive it with fixed timestamps and no clock is read internally. Day and
//! hour are derived in UTC in the MVP (a local-timezone refinement is a
//! follow-up); this keeps the kernel dependency-free and trivially testable.
//!
//! Deliberately simple: weighted state, a circadian curve, exponential topic
//! decay, and cooldowns. No neural anything, no over-built planner.

use std::collections::BTreeMap;

use rand::RngExt;
use serde::{Deserialize, Serialize};

use crate::persona_engine::policy::PersonaPolicy;

/// How many recent topics / actions to retain (bounded so state stays small).
const HISTORY_LIMIT: usize = 12;
/// How many local diary summaries to retain.
const MEMORY_LIMIT: usize = 8;
/// Momentum added to a topic when the persona acts on it.
const TOPIC_BUMP: f64 = 1.0;
/// Cooldown applied to a category after acting on it (milliseconds).
const CATEGORY_COOLDOWN_MS: i64 = 90 * 60 * 1000;

/// The evolving behavior state for one persona. Persisted (as JSON) in the
/// encrypted store between runs so continuity survives a restart.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BehaviorState {
    /// The persona policy id this state belongs to.
    pub persona_id: String,
    /// When the kernel last advanced this state, epoch millis.
    pub last_updated: i64,
    /// The routine name active at `last_updated`, or `None` for quiet hours.
    pub current_routine: Option<String>,
    /// Recently-touched topics/categories (most recent last, bounded).
    pub recent_topics: Vec<String>,
    /// Recently-used topic seeds (the persona's own subtopics, most recent last,
    /// bounded). Lets the goal layer walk THROUGH a persona's interests over days
    /// (fountain pens today, blotting paper tomorrow) instead of repeating one.
    #[serde(default)]
    pub recent_seeds: Vec<String>,
    /// Per-category momentum score (decays over time; higher = hotter).
    pub topic_scores: BTreeMap<String, f64>,
    /// Curiosity drive in `[0, 1]` (appetite for a new topic).
    pub curiosity: f64,
    /// Boredom in `[0, 1]` (rises when recent activity is repetitive).
    pub boredom: f64,
    /// Energy in `[0, 1]` (circadian; low overnight, higher in the day/evening).
    pub energy: f64,
    /// Recent action descriptors (most recent last, bounded).
    pub last_actions: Vec<String>,
    /// Cooldowns: key (e.g. `category:CRAFTS`, `engine:google`) -> epoch millis
    /// until which the key is on cooldown.
    pub cooldowns: BTreeMap<String, i64>,
    /// Need/motive levels (the Sims-like decaying drives). Additive; defaults to
    /// empty for state written before needs existed.
    #[serde(default)]
    pub needs: crate::persona_engine::needs::NeedState,
    /// The world-model: the memory stream and the living interest graph (the
    /// persona's evolving "awareness with meaning"). Additive; a state written
    /// before the world-model existed gets an empty one, seeded on first
    /// [`advance`](BehaviorKernel::advance) from the policy's authored seeds.
    #[serde(default)]
    pub world: crate::persona_engine::world::WorldState,
    /// Local-only persona diary summaries (bounded; decoy-only, no real data).
    /// Superseded by `world.memories` for new content; retained for the
    /// executor's coarse per-session summary line.
    pub memory_summaries: Vec<String>,
    /// Actions performed on `day_epoch` (for the soft daily budget).
    pub actions_today: u32,
    /// The UTC day index `actions_today` is counted for (`now_ms / 86_400_000`).
    pub day_epoch: i64,
}

impl BehaviorState {
    /// A fresh, neutral state for `persona_id`.
    pub fn new(persona_id: impl Into<String>) -> Self {
        Self {
            persona_id: persona_id.into(),
            last_updated: 0,
            current_routine: None,
            recent_topics: Vec::new(),
            recent_seeds: Vec::new(),
            topic_scores: BTreeMap::new(),
            curiosity: 0.5,
            boredom: 0.0,
            energy: 0.5,
            last_actions: Vec::new(),
            cooldowns: BTreeMap::new(),
            needs: crate::persona_engine::needs::NeedState::default(),
            world: crate::persona_engine::world::WorldState::default(),
            memory_summaries: Vec::new(),
            actions_today: 0,
            day_epoch: 0,
        }
    }

    /// The current momentum score for a category (0 when unseen).
    pub fn topic_score(&self, category_name: &str) -> f64 {
        self.topic_scores.get(category_name).copied().unwrap_or(0.0)
    }

    /// Whether `key` is on cooldown at `now`.
    pub fn on_cooldown(&self, key: &str, now: i64) -> bool {
        self.cooldowns.get(key).is_some_and(|&until| until > now)
    }

    /// Record that the persona acted on `category_name` (pursuing topic `seed`)
    /// at `now`: bump the category's momentum, push history (category + seed),
    /// set a category cooldown, and count it toward the daily budget. Called by
    /// the executor after a dispatched action.
    pub fn record_action(&mut self, category_name: &str, seed: &str, now: i64) {
        *self
            .topic_scores
            .entry(category_name.to_string())
            .or_insert(0.0) += TOPIC_BUMP;
        push_bounded(
            &mut self.recent_topics,
            category_name.to_string(),
            HISTORY_LIMIT,
        );
        if !seed.is_empty() {
            push_bounded(&mut self.recent_seeds, seed.to_string(), HISTORY_LIMIT);
            self.world.reinforce(seed, now);
        }
        push_bounded(
            &mut self.last_actions,
            format!("{category_name}:{seed}"),
            HISTORY_LIMIT,
        );
        self.cooldowns.insert(
            format!("category:{category_name}"),
            now.saturating_add(CATEGORY_COOLDOWN_MS),
        );
        let day = day_index(now);
        if day != self.day_epoch {
            self.day_epoch = day;
            self.actions_today = 0;
        }
        self.actions_today = self.actions_today.saturating_add(1);
    }

    /// Append a local-only diary summary (bounded, decoy-only).
    pub fn add_memory_summary(&mut self, summary: impl Into<String>) {
        push_bounded(&mut self.memory_summaries, summary.into(), MEMORY_LIMIT);
    }
}

/// The context resolved for one tick: the day/hour class and the active routine.
#[derive(Debug, Clone, PartialEq)]
pub struct TickContext {
    /// Whether `now` falls on a weekend (UTC).
    pub is_weekend: bool,
    /// The local (UTC in the MVP) hour, 0..=23.
    pub hour: u8,
    /// The active routine name, or `None` during quiet hours.
    pub routine: Option<String>,
    /// Whether this tick is quiet hours (no routine covers it).
    pub quiet_hours: bool,
}

/// The behavior kernel. Stateless; advances a [`BehaviorState`] in place.
#[derive(Debug, Clone, Copy, Default)]
pub struct BehaviorKernel;

impl BehaviorKernel {
    /// Advance `state` to `now`: decay topic momentum, prune expired cooldowns,
    /// recompute the circadian energy and the boredom/curiosity signals, and
    /// resolve the current routine. Returns the resolved [`TickContext`].
    ///
    /// Pure and deterministic given `state`, `policy`, and `now`.
    pub fn advance(
        &self,
        state: &mut BehaviorState,
        policy: &PersonaPolicy,
        now: i64,
    ) -> TickContext {
        let (is_weekend, hour) = day_and_hour(now);
        let elapsed_hours = if state.last_updated > 0 && now > state.last_updated {
            (now - state.last_updated) as f64 / 3_600_000.0
        } else {
            0.0
        };

        // Decay topic momentum by the elapsed time since the last advance.
        if elapsed_hours > 0.0 {
            let half_life = policy.topic_decay.half_life_hours.max(0.01);
            let factor = 0.5_f64.powf(elapsed_hours / half_life);
            for score in state.topic_scores.values_mut() {
                *score *= factor;
            }
            // Drop scores that have decayed to noise so the map stays small.
            state.topic_scores.retain(|_, v| *v >= 0.01);
        }

        // Needs deplete over time: seed any declared need, then drain each by its
        // domain's rate. The most-deficient needs pull behavior in the goal layer.
        let domains = policy.effective_domains();
        state.needs.ensure(domains.iter().map(|d| d.name.clone()));
        for domain in &domains {
            state
                .needs
                .deplete(&domain.name, domain.decay_per_hour, elapsed_hours);
        }

        // Prune expired cooldowns.
        state.cooldowns.retain(|_, &mut until| until > now);

        // Reset the daily counter on a new UTC day.
        let day = day_index(now);
        if day != state.day_epoch {
            state.day_epoch = day;
            state.actions_today = 0;
        }

        // World-model: seed the interest graph on first use (a no-op after),
        // then run daily reflection (decay/prune discovered interests, reinforce,
        // synthesize an insight memory) at most once per UTC day.
        state.world.ensure_seeded(policy, now);
        if day != state.world.last_reflected_day {
            let world_elapsed_hours = if state.world.last_reflected_day > 0 {
                ((day - state.world.last_reflected_day).max(0) as f64) * 24.0
            } else {
                0.0
            };
            state.world.reflect(now, world_elapsed_hours);
            state.world.last_reflected_day = day;
        }

        // Circadian energy.
        state.energy = circadian_energy(hour);

        // Boredom rises with recent repetition; curiosity is its complement,
        // lifted a little by high energy (a lively persona explores more).
        state.boredom = repetition_ratio(&state.recent_topics);
        state.curiosity = ((1.0 - state.boredom) * 0.7 + state.energy * 0.3).clamp(0.05, 0.95);

        let routine = policy.routine_for(is_weekend, hour).map(|r| r.name.clone());
        let quiet_hours = routine.is_none();

        state.current_routine = routine.clone();
        state.last_updated = now;

        TickContext {
            is_weekend,
            hour,
            routine,
            quiet_hours,
        }
    }
}

/// Push `item` onto `history`, keeping at most `limit` entries (oldest dropped).
fn push_bounded(history: &mut Vec<String>, item: String, limit: usize) {
    history.push(item);
    if history.len() > limit {
        let overflow = history.len() - limit;
        history.drain(0..overflow);
    }
}

/// The fraction of the recent history that repeats its single most common entry,
/// in `[0, 1]`. An empty history is not bored.
fn repetition_ratio(recent: &[String]) -> f64 {
    if recent.is_empty() {
        return 0.0;
    }
    let mut counts: BTreeMap<&String, usize> = BTreeMap::new();
    for item in recent {
        *counts.entry(item).or_insert(0) += 1;
    }
    let max = counts.values().copied().max().unwrap_or(1);
    max as f64 / recent.len() as f64
}

/// A coarse circadian energy curve in `[0, 1]`, keyed by hour of day. Low
/// overnight, ramping through the morning, steady in the afternoon, a gentle
/// evening peak, then winding down.
fn circadian_energy(hour: u8) -> f64 {
    match hour {
        0..=5 => 0.15,
        6..=7 => 0.45,
        8..=10 => 0.75,
        11..=13 => 0.7,
        14..=17 => 0.72,
        18..=21 => 0.85,
        22 => 0.5,
        _ => 0.25,
    }
}

/// A jittered inter-arrival delay (seconds) until the persona's NEXT action, so
/// the cadence is never metronomic (a regular clock tick is itself a
/// fingerprint). Higher energy shortens the mean gap (a livelier persona acts
/// more often); the gap is then drawn from an exponential (Poisson-like)
/// distribution via `-ln(1 - u)`, mirroring the household scheduler's model, and
/// clamped to a sane band. Deterministic given `rng`.
pub fn next_delay_seconds(energy: f64, rng: &mut impl RngExt) -> u64 {
    // Mean gap: ~15 min at full energy, ~75 min when flat.
    let e = energy.clamp(0.0, 1.0);
    let mean_secs = 4500.0 - 3600.0 * e;
    let u = rng.random::<f64>().clamp(1e-9, 1.0 - 1e-9);
    let sample = -(1.0 - u).ln() * mean_secs;
    sample.clamp(60.0, 4.0 * 3600.0) as u64
}

/// The UTC day index for `now` (millis since the epoch, floored to days). Clamps
/// a pre-epoch timestamp to day 0.
fn day_index(now: i64) -> i64 {
    now.max(0) / 86_400_000
}

/// Derive `(is_weekend, hour)` in UTC from `now` (epoch millis). Unix day 0
/// (1970-01-01) was a Thursday; indexing Monday=0..Sunday=6, that is index 3.
fn day_and_hour(now: i64) -> (bool, u8) {
    let now = now.max(0);
    let days = now / 86_400_000;
    let hour = ((now / 3_600_000) % 24) as u8;
    // Monday=0 .. Sunday=6.
    let dow = ((days + 3) % 7) as u8;
    let is_weekend = dow >= 5;
    (is_weekend, hour)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persona_engine::builtins;

    fn elias() -> PersonaPolicy {
        // The bundled Elias policy is validated elsewhere; unwrap-free load.
        match builtins::get("elias_rickensworth") {
            Ok(p) => p,
            Err(e) => panic!("elias must load: {e}"),
        }
    }

    // 1970-01-01 was a Thursday; +3 days = Sunday (weekend). Pick timestamps by
    // constructing them from day + hour directly.
    fn ts(day: i64, hour: u8) -> i64 {
        day * 86_400_000 + (hour as i64) * 3_600_000
    }

    #[test]
    fn day_and_hour_is_utc_and_correct() {
        // Day 0 = Thursday (weekday), hour extracted correctly.
        let (weekend, hour) = day_and_hour(ts(0, 9));
        assert!(!weekend);
        assert_eq!(hour, 9);
        // Day 2 = Saturday (weekend).
        assert!(day_and_hour(ts(2, 12)).0);
        // Day 3 = Sunday (weekend).
        assert!(day_and_hour(ts(3, 12)).0);
        // Day 4 = Monday (weekday).
        assert!(!day_and_hour(ts(4, 12)).0);
    }

    #[test]
    fn quiet_hours_resolve_to_no_routine() {
        let policy = elias();
        let kernel = BehaviorKernel;
        let mut state = BehaviorState::new("elias_rickensworth");
        // Day 4 is a Monday; 03:00 is outside every routine window.
        let tick = kernel.advance(&mut state, &policy, ts(4, 3));
        assert!(tick.quiet_hours);
        assert!(tick.routine.is_none());
        assert!(state.current_routine.is_none());
    }

    #[test]
    fn weekday_evening_routine_is_selected() {
        let policy = elias();
        let kernel = BehaviorKernel;
        let mut state = BehaviorState::new("elias_rickensworth");
        // Day 4 Monday, 20:00 -> weekday_evening.
        let tick = kernel.advance(&mut state, &policy, ts(4, 20));
        assert_eq!(tick.routine.as_deref(), Some("weekday_evening"));
        assert!(!tick.quiet_hours);
    }

    #[test]
    fn topic_momentum_decays_with_time() {
        let policy = elias();
        let kernel = BehaviorKernel;
        let mut state = BehaviorState::new("elias_rickensworth");
        kernel.advance(&mut state, &policy, ts(4, 18));
        state.record_action("CRAFTS", "fountain pens", ts(4, 18));
        let hot = state.topic_score("CRAFTS");
        assert!(hot > 0.0);
        // Advance ~2 half-lives (72h) later: momentum should be ~1/4.
        kernel.advance(&mut state, &policy, ts(4, 18) + 72 * 3_600_000);
        let cooled = state.topic_score("CRAFTS");
        assert!(cooled < hot);
        assert!(
            cooled <= hot * 0.30,
            "expected ~1/4 decay, got {cooled} from {hot}"
        );
    }

    #[test]
    fn cooldown_is_set_and_expires() {
        let policy = elias();
        let kernel = BehaviorKernel;
        let mut state = BehaviorState::new("elias_rickensworth");
        let t = ts(4, 18);
        kernel.advance(&mut state, &policy, t);
        state.record_action("CRAFTS", "pens", t);
        assert!(state.on_cooldown("category:CRAFTS", t));
        // After the cooldown window, a later advance prunes it.
        let later = t + CATEGORY_COOLDOWN_MS + 1;
        kernel.advance(&mut state, &policy, later);
        assert!(!state.on_cooldown("category:CRAFTS", later));
    }

    #[test]
    fn energy_follows_the_circadian_curve() {
        let policy = elias();
        let kernel = BehaviorKernel;
        let mut state = BehaviorState::new("elias_rickensworth");
        kernel.advance(&mut state, &policy, ts(4, 3));
        let night = state.energy;
        kernel.advance(&mut state, &policy, ts(4, 20));
        let evening = state.energy;
        assert!(evening > night);
    }

    #[test]
    fn next_delay_is_bounded_and_shortens_with_energy() {
        use rand::rngs::StdRng;
        use rand::SeedableRng;
        let mut rng = StdRng::seed_from_u64(1);
        let d = next_delay_seconds(0.9, &mut rng);
        assert!((60..=4 * 3600).contains(&d), "delay out of band: {d}");
        // Averaged over many draws, high energy gives a shorter mean gap.
        let mean = |e: f64| {
            let mut r = StdRng::seed_from_u64(42);
            let mut total = 0u64;
            for _ in 0..500 {
                total += next_delay_seconds(e, &mut r);
            }
            total / 500
        };
        assert!(
            mean(0.9) < mean(0.1),
            "livelier persona should act more often"
        );
    }

    #[test]
    fn needs_are_seeded_and_deplete_over_time() {
        let policy = elias();
        let kernel = BehaviorKernel;
        let mut state = BehaviorState::new("elias_rickensworth");
        kernel.advance(&mut state, &policy, ts(4, 6));
        // Every declared domain's need is seeded.
        assert!(state.needs.levels.contains_key("hobby"));
        assert!(state.needs.levels.contains_key("wellbeing"));
        let start = state.needs.level("hobby");
        // Ten hours later the hobby need has drained (no action satisfied it).
        kernel.advance(&mut state, &policy, ts(4, 16));
        assert!(state.needs.level("hobby") < start);
    }

    #[test]
    fn daily_counter_resets_on_a_new_day() {
        let policy = elias();
        let kernel = BehaviorKernel;
        let mut state = BehaviorState::new("elias_rickensworth");
        let t = ts(4, 18);
        kernel.advance(&mut state, &policy, t);
        state.record_action("CRAFTS", "pens", t);
        assert_eq!(state.actions_today, 1);
        // Next day, the kernel resets the counter.
        kernel.advance(&mut state, &policy, ts(5, 18));
        assert_eq!(state.actions_today, 0);
    }

    #[test]
    fn boredom_rises_with_repetition() {
        let mut state = BehaviorState::new("p");
        for _ in 0..6 {
            push_bounded(
                &mut state.recent_topics,
                "CRAFTS".to_string(),
                HISTORY_LIMIT,
            );
        }
        assert!((repetition_ratio(&state.recent_topics) - 1.0).abs() < 1e-9);
        state.recent_topics.clear();
        push_bounded(
            &mut state.recent_topics,
            "CRAFTS".to_string(),
            HISTORY_LIMIT,
        );
        push_bounded(
            &mut state.recent_topics,
            "HISTORY".to_string(),
            HISTORY_LIMIT,
        );
        assert!((repetition_ratio(&state.recent_topics) - 0.5).abs() < 1e-9);
    }
}
