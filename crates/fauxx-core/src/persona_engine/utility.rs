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

//! Utility-AI scoring for the goal layer.
//!
//! This is the small "Sims-like" decision core. Each candidate category is
//! scored from a handful of NORMALIZED considerations (each in `[0, 1]`), which
//! are combined MULTIPLICATIVELY: a category is attractive only when ALL of its
//! considerations are reasonably satisfied (the AND-semantics utility AI is
//! known for, per Dave Mark's "Behavioral Mathematics"). The winner is then
//! drawn with a TEMPERATURE-weighted softmax rather than a hard argmax, so the
//! persona's variety, boring pivots, and mild contradictions EMERGE from the
//! sampling instead of being bolted on with a coin flip.
//!
//! Everything here is pure and deterministic given the inputs and the injected
//! RNG, so it is trivially unit-tested.

use rand::RngExt;

/// The considerations for one candidate category. Each field is expected in
/// `[0, 1]`; [`score`] clamps defensively.
#[derive(Debug, Clone, Copy)]
pub struct Considerations {
    /// How core this category is to the persona (e.g. how many routines favor
    /// it). A persona's signature interests score higher than incidental ones.
    pub affinity: f64,
    /// Normalized topic momentum: "I was enjoying this thread." Continuation.
    pub momentum: f64,
    /// Curiosity-scaled appetite for something cold: `curiosity * (1 - momentum)`.
    pub novelty: f64,
    /// Recent-repetition satiation in `[0, 1]`: how much of the recent history is
    /// already this category. High recency dampens the desire to repeat it.
    pub recency: f64,
    /// Whether the category is currently on cooldown (a near-hard gate).
    pub on_cooldown: bool,
}

/// Baseline appetite every non-cooled category carries, so a cold-but-core
/// interest still has a real chance to be picked.
const BASE_INTEREST: f64 = 0.15;
/// Weight on continuing a warm thread.
const MOMENTUM_WEIGHT: f64 = 0.6;
/// Weight on novelty-seeking.
const NOVELTY_WEIGHT: f64 = 0.5;
/// How strongly recent repetition satiates (dampens) a category.
const SATIATION_WEIGHT: f64 = 0.6;
/// Floor on affinity so a low-affinity category is unlikely, not impossible.
const AFFINITY_FLOOR: f64 = 0.05;

/// Score one candidate in `[0, 1)`-ish (unnormalized; only relative magnitudes
/// matter to [`softmax_choose`]). A category on cooldown scores `0`.
pub fn score(c: &Considerations) -> f64 {
    if c.on_cooldown {
        return 0.0;
    }
    let momentum = clamp01(c.momentum);
    let novelty = clamp01(c.novelty);
    let recency = clamp01(c.recency);
    let affinity = clamp01(c.affinity).max(AFFINITY_FLOOR);

    let interest = BASE_INTEREST + MOMENTUM_WEIGHT * momentum + NOVELTY_WEIGHT * novelty;
    let satiation = (1.0 - SATIATION_WEIGHT * recency).clamp(0.0, 1.0);
    // Multiplicative AND-combine: every factor must be non-trivial to win.
    affinity * interest * satiation
}

/// Draw an index into `scores` weighted by a temperature softmax.
///
/// Higher `temperature` flattens the distribution (more exploratory / more
/// boring pivots); lower sharpens it toward the top scorer (greedy). Returns
/// `None` only when every score is zero or non-finite (e.g. all on cooldown).
/// Deterministic given `rng`.
pub fn softmax_choose(scores: &[f64], temperature: f64, rng: &mut impl RngExt) -> Option<usize> {
    if scores.is_empty() {
        return None;
    }
    // If nothing has any weight, there is nothing to choose.
    if !scores.iter().any(|s| s.is_finite() && *s > 0.0) {
        return None;
    }
    let temp = temperature.max(0.05);
    // Numerically stable softmax: subtract the max before exponentiating.
    let max = scores
        .iter()
        .copied()
        .filter(|s| s.is_finite())
        .fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<f64> = scores
        .iter()
        .map(|s| {
            if s.is_finite() && *s > 0.0 {
                ((s - max) / temp).exp()
            } else {
                0.0
            }
        })
        .collect();
    let total: f64 = weights.iter().sum();
    if total <= 0.0 || !total.is_finite() {
        // Degenerate; fall back to the argmax.
        return argmax(scores);
    }
    let mut pick = rng.random::<f64>() * total;
    for (i, w) in weights.iter().enumerate() {
        pick -= w;
        if pick <= 0.0 {
            return Some(i);
        }
    }
    // Floating-point slack: return the last positive-weight index.
    weights.iter().rposition(|w| *w > 0.0)
}

/// The index of the single highest finite score, or `None`.
pub fn argmax(scores: &[f64]) -> Option<usize> {
    scores
        .iter()
        .enumerate()
        .filter(|(_, s)| s.is_finite())
        .fold(None, |best, (i, &s)| match best {
            Some((_, bs)) if bs >= s => best,
            _ => Some((i, s)),
        })
        .map(|(i, _)| i)
}

/// Clamp a value into `[0, 1]` (treating a NaN as 0).
fn clamp01(v: f64) -> f64 {
    if v.is_nan() {
        0.0
    } else {
        v.clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn con(affinity: f64, momentum: f64, novelty: f64, recency: f64, cool: bool) -> Considerations {
        Considerations {
            affinity,
            momentum,
            novelty,
            recency,
            on_cooldown: cool,
        }
    }

    #[test]
    fn cooldown_scores_zero() {
        assert_eq!(score(&con(1.0, 1.0, 1.0, 0.0, true)), 0.0);
    }

    #[test]
    fn momentum_and_affinity_raise_score() {
        let cold = score(&con(1.0, 0.0, 0.0, 0.0, false));
        let warm = score(&con(1.0, 1.0, 0.0, 0.0, false));
        assert!(warm > cold);
        let low_affinity = score(&con(0.2, 1.0, 0.0, 0.0, false));
        assert!(warm > low_affinity);
    }

    #[test]
    fn recent_repetition_satiates() {
        let fresh = score(&con(1.0, 1.0, 0.0, 0.0, false));
        let stale = score(&con(1.0, 1.0, 0.0, 1.0, false));
        assert!(stale < fresh);
    }

    #[test]
    fn softmax_none_when_all_zero() {
        let mut rng = StdRng::seed_from_u64(1);
        assert_eq!(softmax_choose(&[0.0, 0.0, 0.0], 0.3, &mut rng), None);
        assert_eq!(softmax_choose(&[], 0.3, &mut rng), None);
    }

    #[test]
    fn softmax_is_deterministic_for_a_fixed_seed() {
        let scores = [0.2, 0.5, 0.9, 0.1];
        let mut a = StdRng::seed_from_u64(7);
        let mut b = StdRng::seed_from_u64(7);
        assert_eq!(
            softmax_choose(&scores, 0.3, &mut a),
            softmax_choose(&scores, 0.3, &mut b)
        );
    }

    #[test]
    fn low_temperature_favors_the_top_scorer() {
        // With a sharp (low) temperature, the clear top scorer should win the
        // large majority of draws.
        let scores = [0.1, 0.2, 0.95, 0.15];
        let mut wins = 0;
        for s in 0..400u64 {
            let mut rng = StdRng::seed_from_u64(s);
            if softmax_choose(&scores, 0.08, &mut rng) == Some(2) {
                wins += 1;
            }
        }
        assert!(
            wins > 320,
            "expected the top scorer to dominate, got {wins}/400"
        );
    }

    #[test]
    fn high_temperature_spreads_the_choice() {
        // With a flat (high) temperature, lower scorers get chosen sometimes.
        let scores = [0.3, 0.4, 0.9, 0.35];
        let mut non_top = 0;
        for s in 0..400u64 {
            let mut rng = StdRng::seed_from_u64(s);
            if softmax_choose(&scores, 2.0, &mut rng) != Some(2) {
                non_top += 1;
            }
        }
        assert!(
            non_top > 80,
            "expected exploration under high temperature, got {non_top}/400"
        );
    }

    #[test]
    fn argmax_picks_the_highest() {
        assert_eq!(argmax(&[0.1, 0.9, 0.3]), Some(1));
        assert_eq!(argmax(&[]), None);
    }
}
