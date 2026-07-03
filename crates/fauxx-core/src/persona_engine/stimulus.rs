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

//! Stimuli: candidate observations the persona MIGHT notice.
//!
//! A [`Stimulus`] is raw input to the appraisal step
//! ([`crate::persona_engine::appraise`]); by itself it carries no meaning and
//! has no effect. Two producers feed the pipeline:
//!
//! - offline: [`roll_life_event`] draws from the policy's authored
//!   `[[life_events]]` (Sims-style "whims" - small, character-consistent
//!   happenings), pseudo-randomly and not every tick;
//! - online: the executor builds a `Stimulus` from a dispatched query's own
//!   category/seed (today) or, later, from
//!   [`SemanticAssistant::classify_page_text`](crate::persona_engine::sidecar::SemanticAssistant::classify_page_text)
//!   run over a visited page's text.
//!
//! A stimulus never bypasses the category Venn or the harmful-query blocklist:
//! appraisal enforces both before anything is noticed or adopted.

use rand::RngExt;

use crate::persona_engine::policy::PersonaPolicy;
use crate::persona_engine::sidecar::SemanticAssistant;

/// Where a [`Stimulus`] originated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StimulusSource {
    /// A real-world happening (an authored life event).
    Offline,
    /// Something encountered online (a search result, a visited page).
    Online,
}

/// A candidate observation, not yet appraised for meaning.
#[derive(Debug, Clone, PartialEq)]
pub struct Stimulus {
    /// Where this stimulus came from.
    pub source: StimulusSource,
    /// The observation text, in third person (becomes the memory line if
    /// noticed).
    pub text: String,
    /// The [`CategoryPool`](crate::persona::CategoryPool) name this is
    /// topically about, if any. `None` means need-only (no topical content),
    /// which always clears appraisal's category Venn gate.
    pub category: Option<String>,
    /// A candidate topic seed this stimulus may suggest for adoption.
    pub suggests_seed: Option<String>,
    /// The need (domain name) this is relevant to, if any.
    pub need: Option<String>,
    /// The authored/assigned base weight (higher = more likely to matter).
    pub base_weight: f64,
}

/// Per-tick probability that ANY life event fires at all. Kept LOW: a life
/// event is meant to be a rare, noteworthy happening (roughly once a day, given
/// the executor calls this once per hour of an active window), not an hourly
/// occurrence. A persona with a remarkable happening every hour is himself a
/// fingerprint, and the point of these events is Sims-style whims, not noise.
const LIFE_EVENT_CHANCE: f64 = 0.06;

/// Roll for an offline life event: with probability [`LIFE_EVENT_CHANCE`], draw
/// one of the policy's authored `[[life_events]]` weighted by its `weight`.
/// `None` if the policy declares no life events, or the roll comes up empty.
/// Deterministic given `rng`.
pub fn roll_life_event(policy: &PersonaPolicy, rng: &mut impl RngExt) -> Option<Stimulus> {
    if policy.life_events.is_empty() || rng.random::<f64>() >= LIFE_EVENT_CHANCE {
        return None;
    }
    let total: f64 = policy.life_events.iter().map(|e| e.weight.max(0.0)).sum();
    if total <= 0.0 {
        return None;
    }
    let mut pick = rng.random::<f64>() * total;
    for event in &policy.life_events {
        pick -= event.weight.max(0.0);
        if pick <= 0.0 {
            return Some(from_life_event(event));
        }
    }
    policy.life_events.last().map(from_life_event)
}

fn from_life_event(event: &crate::persona_engine::policy::LifeEvent) -> Stimulus {
    Stimulus {
        source: StimulusSource::Offline,
        text: event.text.clone(),
        category: event.category.clone(),
        suggests_seed: event.suggests_seed.clone(),
        need: event.need.clone(),
        base_weight: event.weight,
    }
}

/// Build an online-discovery stimulus from a dispatched query. The baseline
/// (deterministic-only) source of "what did he encounter online": the
/// category/seed he was already searching, restated as something he might
/// notice more of. When `sidecar` is enabled, it is asked to
/// [`propose_subseed`](SemanticAssistant::propose_subseed) - the one thing the
/// deterministic path structurally cannot do, genuinely invent a NEW topic
/// rather than a new wording of an existing one. The suggestion is carried as
/// `suggests_seed` UNVETTED: appraisal still enforces the category Venn gate
/// and the harmful-query blocklist before it can ever be noticed or adopted.
pub fn from_dispatched_query(
    policy: &PersonaPolicy,
    category: &str,
    query_seed: &str,
    sidecar: &dyn SemanticAssistant,
) -> Stimulus {
    let text = format!("came across more about {query_seed} while searching");
    let suggests_seed = sidecar.propose_subseed(policy, category, &text);
    Stimulus {
        source: StimulusSource::Online,
        text,
        category: Some(category.to_string()),
        suggests_seed,
        need: None,
        base_weight: 1.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    use crate::persona_engine::builtins;
    use crate::persona_engine::policy::LifeEvent;

    /// A minimal policy plus a couple of authored life events, self-contained
    /// so these tests do not depend on how a specific built-in persona (e.g.
    /// Elias) chooses to author its own events.
    fn policy_with_events() -> PersonaPolicy {
        let mut policy = match builtins::get("elias_rickensworth") {
            Ok(p) => p,
            Err(e) => panic!("elias must load: {e}"),
        };
        policy.life_events = vec![
            LifeEvent {
                id: "shed_roof_leak".to_string(),
                text: "a leak appears in the shed roof".to_string(),
                category: Some("HOME_IMPROVEMENT".to_string()),
                suggests_seed: Some("shed roof repair".to_string()),
                need: Some("upkeep".to_string()),
                weight: 1.0,
            },
            LifeEvent {
                id: "swap_meet".to_string(),
                text: "a neighbour mentions a model railway swap meet".to_string(),
                category: Some("OUTDOOR_RECREATION".to_string()),
                suggests_seed: Some("model railway swap meet".to_string()),
                need: Some("hobby".to_string()),
                weight: 1.0,
            },
        ];
        policy
    }

    #[test]
    fn no_life_events_never_fires() {
        let mut policy = policy_with_events();
        policy.life_events.clear();
        let mut rng = StdRng::seed_from_u64(1);
        for _ in 0..50 {
            assert!(roll_life_event(&policy, &mut rng).is_none());
        }
    }

    #[test]
    fn life_events_fire_sometimes_but_not_always() {
        let policy = policy_with_events();
        let mut fired = 0;
        let mut rng = StdRng::seed_from_u64(7);
        for _ in 0..200 {
            if roll_life_event(&policy, &mut rng).is_some() {
                fired += 1;
            }
        }
        assert!(
            fired > 0,
            "expected at least one life event across 200 rolls"
        );
        assert!(
            fired < 200,
            "expected life events to be intermittent, not constant"
        );
    }

    #[test]
    fn rolled_event_is_one_of_the_authored_events() {
        let policy = policy_with_events();
        let mut rng = StdRng::seed_from_u64(3);
        for _ in 0..100 {
            if let Some(stim) = roll_life_event(&policy, &mut rng) {
                assert!(policy.life_events.iter().any(|e| e.text == stim.text));
                assert_eq!(stim.source, StimulusSource::Offline);
            }
        }
    }

    #[test]
    fn dispatched_query_stimulus_with_disabled_sidecar_suggests_nothing() {
        use crate::persona_engine::sidecar::DisabledAssistant;
        let policy = policy_with_events();
        let s = from_dispatched_query(&policy, "CRAFTS", "blue black ink", &DisabledAssistant);
        assert_eq!(s.source, StimulusSource::Online);
        assert_eq!(s.category.as_deref(), Some("CRAFTS"));
        assert!(s.text.contains("blue black ink"));
        assert!(s.suggests_seed.is_none());
    }

    #[test]
    fn dispatched_query_stimulus_carries_sidecar_proposed_seed() {
        struct StubAssistant;
        impl SemanticAssistant for StubAssistant {
            fn is_enabled(&self) -> bool {
                true
            }
            fn propose_subseed(
                &self,
                _persona: &PersonaPolicy,
                _category: &str,
                _context: &str,
            ) -> Option<String> {
                Some("model train weathering paint".to_string())
            }
        }
        let policy = policy_with_events();
        let s = from_dispatched_query(
            &policy,
            "OUTDOOR_RECREATION",
            "garden railways",
            &StubAssistant,
        );
        assert_eq!(
            s.suggests_seed.as_deref(),
            Some("model train weathering paint")
        );
    }
}
