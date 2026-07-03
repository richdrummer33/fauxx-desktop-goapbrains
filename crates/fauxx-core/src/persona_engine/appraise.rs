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

//! Appraisal: the attention/interest filter that turns a raw [`Stimulus`] into
//! MEANING (or nothing at all).
//!
//! This is the crux of the world-model: a stimulus is not automatically
//! remembered or acted on. It is scored for [`Appraisal::salience`] against the
//! persona's fixed identity and current state, in the spirit of appraisal
//! theories of interest and emotion (the OCC model; Silvia's psychology of
//! interest; Loewenstein's information-gap theory of curiosity):
//!
//! ```text
//! salience = value_fit * attention * (relevance + curiosity_gain * novelty + need_pull)
//! ```
//!
//! - `value_fit` is a HARD gate: a stimulus about a category outside the
//!   policy's `allowed_categories` (or inside `forbidden_categories`) scores
//!   ZERO and is never noticed, full stop. This is simultaneously the "Venn
//!   diagram" constraint the persona's interests may specialize within, and a
//!   safety boundary: appraisal cannot drift the persona into forbidden topics.
//! - `attention` models limited bandwidth: tired or highly task-focused
//!   (Conscientiousness) personas notice less background noise.
//! - `relevance` is a cheap stand-in for spreading activation: does this touch
//!   something already in the interest graph or recent activity?
//! - `novelty` is an information-gap signal, scaled by Openness
//!   (`curiosity_gain`): a topic the persona does not already have as an
//!   interest is more likely to catch attention, more so for an open persona.
//! - `need_pull` is the current deficit of the stimulus's relevant need, if any.
//!
//! A salient-enough stimulus is NOTICED (a [`Memory`] is recorded); a
//! sufficiently salient one that suggests an in-Venn, blocklist-clean seed is
//! ADOPTED into the interest graph. Both steps happen in [`ingest`], which is
//! the only writer of discovered interests: nothing here executes anything, and
//! nothing here can add a category outside the policy's Venn.

use crate::persona_engine::kernel::BehaviorState;
use crate::persona_engine::policy::PersonaPolicy;
use crate::persona_engine::stimulus::{Stimulus, StimulusSource};
use crate::persona_engine::world::{Memory, MemoryKind};
use crate::querybank::QueryBlocklist;

/// Salience at/above which a stimulus is NOTICED (recorded as a memory).
const NOTICE_THRESHOLD: f64 = 0.35;
/// Baseline salience at/above which a noticed stimulus's suggested seed may be
/// ADOPTED; nudged down by Openness (see [`appraise`]).
const BASE_ADOPT_THRESHOLD: f64 = 0.55;
/// Floor the Openness-adjusted adopt threshold never drops below (adoption is
/// always at least somewhat selective, however open the persona is).
const MIN_ADOPT_THRESHOLD: f64 = 0.25;
/// Fraction of a stimulus's salience carried into a freshly adopted interest's
/// starting weight (kept small; it must be reinforced by actually being pursued
/// to become a real pull, not just noticed once).
const ADOPTION_WEIGHT_SCALE: f64 = 0.3;

/// The result of appraising one [`Stimulus`].
#[derive(Debug, Clone, PartialEq)]
pub struct Appraisal {
    /// The computed salience (unbounded above zero; typically small, `0..~2`).
    pub salience: f64,
    /// Whether the stimulus cleared the notice threshold (and so is recorded).
    pub noticed: bool,
    /// Whether the stimulus's category passed the hard Venn gate. `false`
    /// forces `salience == 0.0` and `noticed == false`.
    pub value_fit: bool,
    /// The seed to adopt into the interest graph, if this stimulus both
    /// suggested one and cleared the (safety- and blocklist-gated) adopt bar.
    pub adopted_seed: Option<String>,
}

/// Appraise `stimulus` against `policy` (fixed identity) and `state` (current
/// world/needs/mood). Pure; performs no mutation and no I/O.
pub fn appraise(policy: &PersonaPolicy, state: &BehaviorState, stimulus: &Stimulus) -> Appraisal {
    let value_fit = category_value_fit(policy, stimulus.category.as_deref());
    if !value_fit {
        return Appraisal {
            salience: 0.0,
            noticed: false,
            value_fit: false,
            adopted_seed: None,
        };
    }

    let personality = &policy.identity.personality;

    // Attention: limited bandwidth. Low energy or a highly conscientious
    // (task-focused) persona notices less background stimulus.
    let attention =
        ((0.3 + 0.7 * state.energy) * (1.0 - 0.4 * personality.conscientiousness)).clamp(0.05, 1.0);

    let relevance = relevance_score(state, stimulus);
    let novelty = novelty_score(state, stimulus);
    let need_pull = stimulus
        .need
        .as_deref()
        .map(|n| state.needs.deficit(n))
        .unwrap_or(0.0);
    let curiosity_gain = 0.4 + 0.8 * personality.openness;

    let salience =
        (attention * (relevance + curiosity_gain * novelty + need_pull) * stimulus.base_weight)
            .clamp(0.0, 3.0);
    let noticed = salience >= NOTICE_THRESHOLD;

    // A more open persona adopts more readily (a lower bar), but adoption is
    // never automatic or unbounded.
    let adopt_threshold =
        (BASE_ADOPT_THRESHOLD - 0.3 * personality.openness).max(MIN_ADOPT_THRESHOLD);
    let adopted_seed = if noticed && salience >= adopt_threshold {
        stimulus
            .suggests_seed
            .as_deref()
            .filter(|s| !s.trim().is_empty() && !QueryBlocklist::bundled().is_blocked(s))
            .map(str::to_string)
    } else {
        None
    };

    Appraisal {
        salience,
        noticed,
        value_fit: true,
        adopted_seed,
    }
}

/// Ingest an already-appraised stimulus: if noticed, record a [`Memory`]; if
/// adopted, add/reinforce the discovered interest in the world's interest
/// graph. An un-noticed stimulus (below threshold, or failed the Venn gate)
/// leaves no trace, as far as the persona's mind is concerned it never happened.
pub fn ingest(state: &mut BehaviorState, stimulus: &Stimulus, appraisal: &Appraisal, now: i64) {
    if !appraisal.noticed {
        return;
    }
    let kind = match stimulus.source {
        StimulusSource::Offline => MemoryKind::OfflineEvent,
        StimulusSource::Online => MemoryKind::OnlineDiscovery,
    };
    let mut tags = Vec::new();
    if let Some(cat) = &stimulus.category {
        tags.push(cat.clone());
    }
    if let Some(seed) = &stimulus.suggests_seed {
        tags.push(seed.clone());
    }
    state.world.remember(Memory {
        at: now,
        kind,
        text: stimulus.text.clone(),
        salience: appraisal.salience,
        tags,
        last_access: now,
        hits: 0,
    });

    if let (Some(seed), Some(category)) = (&appraisal.adopted_seed, &stimulus.category) {
        state.world.adopt(
            seed,
            category,
            appraisal.salience * ADOPTION_WEIGHT_SCALE,
            now,
        );
    }
}

/// The hard Venn gate: a `None` category (a need-only stimulus, no topical
/// content) always passes; a `Some` category must be in `allowed_categories`
/// and not in `forbidden_categories`.
fn category_value_fit(policy: &PersonaPolicy, category: Option<&str>) -> bool {
    match category {
        None => true,
        Some(cat) => {
            policy.allowed_categories.iter().any(|c| c == cat)
                && !policy.forbidden_categories.iter().any(|c| c == cat)
        }
    }
}

/// A cheap stand-in for spreading activation: does this stimulus touch
/// something already in mind (an existing interest in the category, or recent
/// activity in it)?
fn relevance_score(state: &BehaviorState, stimulus: &Stimulus) -> f64 {
    let Some(category) = &stimulus.category else {
        return 0.1;
    };
    let has_interest = !state.world.interests_for_category(category).is_empty();
    let recent_hits = state
        .recent_topics
        .iter()
        .filter(|t| t.as_str() == category)
        .count()
        .min(3);
    let base = if has_interest { 0.3 } else { 0.0 };
    (base + 0.15 * recent_hits as f64).min(1.0)
}

/// An information-gap stand-in: a suggested seed the persona does not already
/// have in its interest graph is more novel (worth noticing) than one it does.
fn novelty_score(state: &BehaviorState, stimulus: &Stimulus) -> f64 {
    match &stimulus.suggests_seed {
        Some(seed) => {
            let already_known = state.world.interests.iter().any(|i| &i.seed == seed);
            if already_known {
                0.1
            } else {
                0.8
            }
        }
        None => 0.3,
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

    fn seeded_state(policy: &PersonaPolicy) -> BehaviorState {
        let mut state = BehaviorState::new(&policy.id);
        state.world.ensure_seeded(policy, 0);
        state.energy = 0.8;
        state
    }

    #[test]
    fn outside_category_stimulus_is_never_noticed() {
        let policy = elias();
        let state = seeded_state(&policy);
        // FINANCE is not in Elias's allowed categories.
        let stim = Stimulus {
            source: StimulusSource::Offline,
            text: "a stock tip from a neighbour".to_string(),
            category: Some("FINANCE".to_string()),
            suggests_seed: Some("day trading tips".to_string()),
            need: None,
            base_weight: 5.0,
        };
        let a = appraise(&policy, &state, &stim);
        assert!(!a.value_fit);
        assert_eq!(a.salience, 0.0);
        assert!(!a.noticed);
        assert!(a.adopted_seed.is_none());
    }

    #[test]
    fn forbidden_category_stimulus_is_never_noticed() {
        let policy = elias();
        let state = seeded_state(&policy);
        // MEDICAL is explicitly forbidden for Elias even though weight is high.
        let stim = Stimulus {
            source: StimulusSource::Offline,
            text: "a health scare in the news".to_string(),
            category: Some("MEDICAL".to_string()),
            suggests_seed: None,
            need: Some("wellbeing".to_string()),
            base_weight: 10.0,
        };
        let a = appraise(&policy, &state, &stim);
        assert!(!a.value_fit);
        assert!(!a.noticed);
    }

    #[test]
    fn in_venn_novel_relevant_stimulus_is_noticed_and_adopted() {
        let policy = elias();
        let state = seeded_state(&policy);
        let stim = Stimulus {
            source: StimulusSource::Offline,
            text: "a neighbour mentions weathering paint for model trains".to_string(),
            category: Some("OUTDOOR_RECREATION".to_string()),
            suggests_seed: Some("model train weathering paint".to_string()),
            need: Some("hobby".to_string()),
            base_weight: 2.0,
        };
        let a = appraise(&policy, &state, &stim);
        assert!(a.value_fit);
        assert!(a.noticed, "salience {}", a.salience);
        assert_eq!(
            a.adopted_seed.as_deref(),
            Some("model train weathering paint")
        );
    }

    #[test]
    fn unsafe_suggested_seed_is_never_adopted_even_if_noticed() {
        let policy = elias();
        let state = seeded_state(&policy);
        let stim = Stimulus {
            source: StimulusSource::Offline,
            text: "a strange rumor at the pub".to_string(),
            category: Some("OUTDOOR_RECREATION".to_string()),
            suggests_seed: Some("call 988 now".to_string()), // blocklisted pattern
            need: Some("hobby".to_string()),
            base_weight: 2.0,
        };
        let a = appraise(&policy, &state, &stim);
        assert!(a.adopted_seed.is_none());
    }

    #[test]
    fn ingest_records_memory_only_when_noticed() {
        let policy = elias();
        let mut state = seeded_state(&policy);
        let stim = Stimulus {
            source: StimulusSource::Offline,
            text: "nothing much happens".to_string(),
            category: Some("OUTDOOR_RECREATION".to_string()),
            suggests_seed: None,
            need: None,
            base_weight: 0.01,
        };
        let a = appraise(&policy, &state, &stim);
        assert!(!a.noticed, "expected low-weight stimulus to go unnoticed");
        ingest(&mut state, &stim, &a, 1000);
        assert!(state.world.memories.is_empty());
    }

    #[test]
    fn ingest_adopts_seed_into_the_interest_graph() {
        let policy = elias();
        let mut state = seeded_state(&policy);
        let before = state.world.interests.len();
        let stim = Stimulus {
            source: StimulusSource::Offline,
            text: "a neighbour mentions weathering paint for model trains".to_string(),
            category: Some("OUTDOOR_RECREATION".to_string()),
            suggests_seed: Some("model train weathering paint".to_string()),
            need: Some("hobby".to_string()),
            base_weight: 2.0,
        };
        let a = appraise(&policy, &state, &stim);
        ingest(&mut state, &stim, &a, 1000);
        assert_eq!(state.world.interests.len(), before + 1);
        assert!(!state.world.memories.is_empty());
        let adopted = state
            .world
            .interests
            .iter()
            .find(|i| i.seed == "model train weathering paint");
        assert!(adopted.is_some());
    }

    #[test]
    fn higher_openness_lowers_the_adoption_bar() {
        let mut open_policy = elias();
        open_policy.identity.personality.openness = 0.95;
        let mut closed_policy = elias();
        closed_policy.identity.personality.openness = 0.05;

        let state_open = seeded_state(&open_policy);
        let state_closed = seeded_state(&closed_policy);

        // A moderately salient, previously-unknown stimulus: open Elias adopts,
        // closed Elias (higher bar) does not.
        let stim = Stimulus {
            source: StimulusSource::Offline,
            text: "a curious flyer for a niche craft fair".to_string(),
            category: Some("CRAFTS".to_string()),
            suggests_seed: Some("marbled paper making".to_string()),
            need: None,
            base_weight: 0.6,
        };
        let a_open = appraise(&open_policy, &state_open, &stim);
        let a_closed = appraise(&closed_policy, &state_closed, &stim);
        assert!(a_open.salience > a_closed.salience);
    }

    #[test]
    fn higher_conscientiousness_lowers_attention_and_salience() {
        let mut focused = elias();
        focused.identity.personality.conscientiousness = 0.95;
        let mut relaxed = elias();
        relaxed.identity.personality.conscientiousness = 0.05;

        let state_focused = seeded_state(&focused);
        let state_relaxed = seeded_state(&relaxed);
        let stim = Stimulus {
            source: StimulusSource::Offline,
            text: "something catches the eye".to_string(),
            category: Some("HISTORY".to_string()),
            suggests_seed: None,
            need: None,
            base_weight: 1.0,
        };
        let a_focused = appraise(&focused, &state_focused, &stim);
        let a_relaxed = appraise(&relaxed, &state_relaxed, &stim);
        assert!(a_relaxed.salience > a_focused.salience);
    }
}
