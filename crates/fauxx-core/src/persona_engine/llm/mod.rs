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

//! The optional local LLM layer: a bounded AUGMENTATION of the deterministic
//! world-model, never a replacement for it.
//!
//! `fauxx-core` ships with NO cloud provider and NO bundled model. This module
//! talks to a LOCAL LLM server (LM Studio's OpenAI-compatible endpoint) over
//! plain loopback HTTP, and only when explicitly enabled via [`lmstudio::LlmConfig`].
//! It implements [`crate::persona_engine::sidecar::SemanticAssistant`], so it
//! plugs into exactly the same seams `DisabledAssistant` does, with the same
//! fail-closed contract: a disabled config, a network error, a timeout, or a
//! malformed/out-of-schema response all degrade to "unavailable", identical to
//! the sidecar being off. It can be safely excluded from a build's runtime
//! behavior entirely by simply never enabling it; nothing about the
//! deterministic pipeline depends on it being present.

pub mod lmstudio;

pub use lmstudio::{LlmConfig, LlmTransport, LmStudioAssistant, LmStudioTransport, MockTransport};
