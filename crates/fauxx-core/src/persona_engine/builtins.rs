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

//! Built-in persona policies, bundled at compile time.
//!
//! Personas ship as TOML embedded via `include_str!` (the same
//! compile-time-bundled-dataset convention the query banks and broker registry
//! use), so `persona-engine list`/`show` work with no files on disk. The CLI can
//! still load an external policy file by path for `validate`.

use crate::persona_engine::policy::PersonaPolicy;

/// The bundled Elias Rickensworth policy TOML.
pub const ELIAS_TOML: &str = include_str!("personas/elias_rickensworth.toml");

/// `(id, toml)` for every built-in persona policy.
const BUILTINS: &[(&str, &str)] = &[("elias_rickensworth", ELIAS_TOML)];

/// The ids of the built-in personas, in declaration order.
pub fn list() -> Vec<&'static str> {
    BUILTINS.iter().map(|(id, _)| *id).collect()
}

/// Whether `id` names a built-in persona.
pub fn is_builtin(id: &str) -> bool {
    BUILTINS.iter().any(|(bid, _)| *bid == id)
}

/// Parse and return the built-in policy named `id`, or a
/// [`CoreError::PersonaEngine`](crate::error::CoreError::PersonaEngine) if there
/// is no such built-in. A built-in whose bundled TOML fails to parse is a build
/// error surfaced by [`builtins_parse`](tests) in CI.
pub fn get(id: &str) -> crate::Result<PersonaPolicy> {
    let toml = BUILTINS
        .iter()
        .find(|(bid, _)| *bid == id)
        .map(|(_, toml)| *toml)
        .ok_or_else(|| {
            crate::CoreError::PersonaEngine(format!("no built-in persona policy named {id:?}"))
        })?;
    PersonaPolicy::from_toml_str(toml)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn elias_is_listed() {
        assert!(list().contains(&"elias_rickensworth"));
        assert!(is_builtin("elias_rickensworth"));
        assert!(!is_builtin("nobody"));
    }

    #[test]
    fn every_builtin_parses_and_validates() -> crate::Result<()> {
        // The bundled personas must parse AND be free of policy issues, so a
        // shipped built-in is never subtly broken.
        for id in list() {
            let policy = get(id)?;
            let issues = policy.validate();
            assert!(issues.is_empty(), "built-in {id} has issues: {issues:?}");
        }
        Ok(())
    }

    #[test]
    fn elias_maps_hobbies_onto_frozen_categories() -> crate::Result<()> {
        let elias = get("elias_rickensworth")?;
        assert_eq!(elias.display_name, "Elias Rickensworth");
        assert!(!elias.fiction_notice.trim().is_empty());
        // His backing persona interests are all real CategoryPool values.
        let cats = elias.backing_persona.interests;
        for c in [
            "HISTORY",
            "CRAFTS",
            "HOME_IMPROVEMENT",
            "OUTDOOR_RECREATION",
        ] {
            assert!(cats.iter().any(|i| i == c), "missing backing interest {c}");
        }
        // And he carries fountain-pen flavor under CRAFTS.
        assert!(elias
            .topic_seeds
            .get("CRAFTS")
            .is_some_and(|seeds| seeds.iter().any(|s| s.contains("fountain pen"))));
        Ok(())
    }

    #[test]
    fn unknown_builtin_is_an_error() {
        assert!(get("does_not_exist").is_err());
    }
}
