# Persona Engine (decoy behavior layer)

The persona engine is "The Sims for privacy decoy personas", but bounded and
safety-constrained. It sits on top of the existing decoy machinery (the query
generator, the harmful-query blocklist, the isolated Chromium decoy browser) and
adds a small control brain: a persona decides, deterministically, what boring
thing to search for right now, and the LLM never gets to drive.

It is NOT an autonomous LLM browser agent. A deterministic goal/control layer
chooses behavior; an optional LLM sidecar is a narrow, schema-validated helper
that is disabled by default and ships with no model in the MVP.

## The pipeline

```
Persona Policy -> Behavior Kernel -> Sense/Appraise/Ingest -> Goal Layer
    -> Planner -> (optional) LLM Sidecar -> Safety Gate -> Executor -> Logs
```

- Persona Policy (`persona_engine::policy`): a TOML document that declares who a
  persona is allowed to be: identity, allowed and forbidden topic categories,
  free-text topic seeds (the flavor), routines, activity budget, and hard safety
  restrictions. Built-in personas are bundled at compile time; external policy
  files can be validated by path.
- Behavior Kernel (`persona_engine::kernel`): a small Sims-like state model
  (time of day, weekday/weekend, energy on a circadian curve, curiosity,
  boredom, per-topic momentum with exponential decay, cooldowns). Pure and
  deterministic given the wall-clock time.
- Sense / Appraise / Ingest (`persona_engine::stimulus`, `persona_engine::
  appraise`, `persona_engine::world`): the persona's evolving world-model. See
  below; this is what lets interests specialize over time while the persona's
  fixed identity never moves.
- Goal Layer (`persona_engine::goal`): the control plane. It scores the routine's
  candidate categories with the utility model (see below) and draws ONE with a
  temperature softmax, then fills in a goal: routine, goal type, action type,
  category, subcategory, budget, allowed modules, and a boring human reason.
- Planner (`persona_engine::planner`): turns the goal into structured candidate
  intents carrying the actual search query. Rules-first and deterministic; it
  never emits an arbitrary URL, and it keeps the persona IN CHARACTER (see below).
- LLM Sidecar (`persona_engine::sidecar`, optionally backed by
  `persona_engine::llm::LmStudioAssistant`): an optional, opt-in helper. See
  limits below.
- Safety Gate (`persona_engine::safety`): mandatory, deterministic, fail-closed
  enforcement. Every intent must pass before it can execute.
- Executor: reuses `browser::search::dispatch_planned_queries`, which drives the
  isolated decoy browser through the same R3 auth-flow and HTTPS guardrail as
  every other decoy visit.
- Logs (`persona_engine::log`): one decoy-only JSONL record per planned or
  executed action, persisted in the encrypted store and exportable to a file.

## Elias mode

The first built-in persona is Elias Rickensworth, a fictional, harmless,
oddly specific benign hobbyist (Victorian railway lamps, fountain pens, blue
black ink, wool waistcoats, rainwater barrels, antique barometers, garden
railways, used books, brass hinges). He is not a real person and must not
impersonate one; his policy carries an explicit fiction notice.

Elias's eccentric hobbies are mapped onto the closest of the 32 frozen
`CategoryPool` values (`HISTORY`, `CRAFTS`, `HOME_IMPROVEMENT`,
`OUTDOOR_RECREATION`, plus `SCIENCE` for weather and barometer trivia), so the
existing query generator and decoy browser can execute for him without touching
the frozen cross-device persona wire model. His routines cover a weekday morning
(weather and tea), midday (household and tools), evening (hobby browsing), and a
longer weekend session. Topic decay, occasional pivots, and skipped days keep him
from being perfectly predictable.

## Decision model (utility AI)

The goal layer is a small utility-AI selector, the same family of technique The
Sims uses (`persona_engine::utility`). Each candidate category is scored from a
handful of considerations, each normalized to zero-to-one:

- affinity: how core the category is to the persona, derived from how many
  routines favor it (a signature interest scores higher than an incidental one),
- momentum: continuing a warm thread the persona has been enjoying,
- novelty: a curiosity-scaled appetite for a cold topic,
- satiation: a penalty for having done this category a lot recently,
- cooldown: a near-hard gate right after acting on a category.

These are combined MULTIPLICATIVELY (a category is attractive only when all of
its considerations are decent, per Dave Mark's utility-AI "Behavioral
Mathematics"), and the winner is drawn with a temperature-weighted softmax rather
than a hard argmax. Variety, boring pivots, and mild contradictions therefore
EMERGE from the sampling instead of being bolted on with a coin flip. A higher
temperature (nudged up by the policy's `pivot_probability`) makes the persona
more restless. Skipping a whole run (an idle day) stays a separate draw.

## In-character queries and interest threading

The persona picks the category AND a topic seed; the queries should sound like
the persona, not like the broad category corpus. The planner builds candidates in
order: (1) the chosen seed in the persona's own words, (2) on-topic refinements of
it, (3) the persona's OTHER curated seeds for that category, and only (4) a light
top-up from the generic query bank if the persona is seed-poor. So Elias searches
"antique barometers" and "how to read a falling barometer", not whatever a generic
history corpus happens to contain.

Interest threading: the goal layer prefers a seed the persona has NOT used
recently (tracked in `recent_seeds`), so consecutive sessions walk through the
persona's interests rather than repeating one. On top of that, a policy can
declare authored narrative arcs (`seed_followups`): after pursuing a seed, the
next seed in the same category is biased toward its follow-ups, so an interest
unfolds as a multi-day thread (fountain pens, then blotting paper, then cheap
paper). Every candidate, from any source, still passes the harmful-query
blocklist and the Safety Gate.

Cadence jitter: each plan carries a `suggested_next_delay_seconds`, an
exponential (Poisson-like) inter-arrival scaled by energy, so a driver schedules
a non-metronomic cadence (a perfectly regular clock tick is itself a
fingerprint).

## Needs, domains, and a life beyond hobbies

A persona is a person, not just a hobby. The engine models a small vector of
decaying needs/motives (`persona_engine::needs`), the same idea The Sims uses:
time depletes each need, actions satisfy it, and the goal layer services
whichever need is most deficient (weighted, not greedy).

Each need is declared by a `domain` in the policy. A domain says which need it
serves and HOW:

- an ONLINE domain (categories plus a high `online_bias`) is satisfied by decoy
  searches (Elias's `hobby` and `upkeep`),
- an OFFLINE domain (no categories, or a low `online_bias`) is satisfied in the
  real world and emits NOTHING on the wire (Elias's `wellbeing` and `errands`:
  he goes for a walk, potters in the shed, walks to the chemist).

Selection is desire then goal then action: the deficit picks a domain (a
DESIRE), the utility model picks a category within it (a GOAL), and the planner
produces the query (the ACTION). This is deliberately GOAP-adjacent without the
A* machinery, so richer multi-step planning can slot in later.

Offline domains are also a safety feature, not just realism: a sensitive
real-life need (health) is modeled as offline, so the decoy never emits medical
or other self-signalling queries on the persona's behalf. Domains are additive;
a policy that declares none behaves exactly as before (one synthesized online
`hobby` domain).

## The world-model: memory, appraisal, and adaptive interests

A persona's interests should not be a fixed shopping list forever. A real person
notices things (something a neighbour says, a flyer, a page they read), holds
onto the ones that matter to them, and lets those reshape what they pursue,
while who they fundamentally ARE never changes. The persona engine models this
with three small, deterministic pieces (`persona_engine::stimulus`,
`persona_engine::appraise`, `persona_engine::world`), drawing on the Generative
Agents memory-stream/reflection design (Park et al. 2023), ACT-R's cheap
recency+frequency activation for retrieval and forgetting, BDI's split between
fixed desires and a mutable world-model, and appraisal theories of interest and
curiosity (the OCC model; Loewenstein's information-gap theory).

**Identity is fixed and gravitational.** A policy's `[identity]` block declares
`core_values` (flavor) and a Big Five (OCEAN) `personality` vector, each trait
`0..1`. This NEVER changes at runtime. It only parameterizes the machinery
below: Openness raises curiosity's weight in appraisal and lowers the bar to
adopt a new interest (a more open persona specializes more readily, without
wandering outside its Venn); Conscientiousness sharpens attention (a more
task-focused persona notices less background stimulus). Elias is moderately
open, highly conscientious, introverted, and placid.

**Stimuli are candidate observations.** Offline, a policy's authored
`[[life_events]]` are small, character-consistent "whims" (Sims-style: "a
neighbour mentions a model railway swap meet") that fire rarely, not every tick.
Online, a dispatched query itself becomes a stimulus (`stimulus::
from_dispatched_query`); a future LLM sidecar seam (`classify_page_text`) can
extract genuinely new candidate topics from a visited page's real text.

**Appraisal decides what MATTERS.** Every stimulus is scored:

```
salience = value_fit * attention * (relevance + curiosity_gain * novelty + need_pull)
```

`value_fit` is a HARD gate: a stimulus about a category outside the policy's
`allowed_categories` (or inside `forbidden_categories`) scores zero and is never
noticed, full stop. This is simultaneously the Venn-diagram constraint interests
may specialize WITHIN, and a safety boundary appraisal cannot cross. Above a
notice threshold, the stimulus is recorded as a `Memory`; above a (Openness-
adjusted, always blocklist- and Venn-gated) adopt threshold, its suggested seed
is adopted into the interest graph.

**The interest graph replaces the flat seed list at runtime.** Authored seeds
(from `topic_seeds`) are permanent gravity wells that never decay below their
floor. A newly adopted, discovered interest starts light and DECAYS if never
reinforced; if it keeps getting noticed or pursued, its weight climbs and it
starts winning seed selection within its category, same as a well-worn authored
one. `choose_seed` (in `goal.rs`) draws from this graph, weighted, so
specialization is a real, gradual, reversible pull, not a one-shot switch.

**Reflection runs once a day**, folded into the kernel's routine advance:
discovered interests decay (authored ones are untouched), anything below a
floor is pruned, and if one tag dominates recent noticed memories, a synthesized
insight memory is added (`"keeps returning to CRAFTS lately"`). This is the
deterministic stand-in for an LLM writing genuine reflective insight.

None of this executes anything by itself. A discovered interest is just another
seed the (unchanged) planner and Safety Gate treat exactly like an authored one:
blocklist-gated, category-Venn-gated, budget-capped. `persona-engine plan`
shows the `sensed:` line (what fired this tick, its salience, whether it was
noticed, and any adopted seed) and an `interests:` count (total / discovered),
so the whole loop is inspectable in dry-run.

## The two identity models

`SyntheticPersona` is the frozen cross-device wire contract shared with the phone
(a UUID and 32 fixed categories). The persona engine does not change it. A
`PersonaPolicy` is a separate, desktop-local behavioral layer that references a
`backing_persona` block; on a live `run-once` the engine materializes a
`SyntheticPersona` (with a deterministic id derived from the policy id) into the
store and drives its isolated decoy browser. The policy is the brain; the frozen
persona is the body.

## LLM sidecar limits

The sidecar is a helper, never a driver. It MUST NOT choose URLs, pick action
types or modules, decide what the persona wants, escalate volume or risk, or
bypass the Safety Gate. At most it phrases an already-approved query seed into
candidate queries, summarizes prior activity into a local diary line, classifies
page text, proposes ONE emergent sub-interest near an existing gravity well, or
nudges the deterministic appraisal salience up or down.

Rules enforced in code:

- Disabled by default. `DisabledAssistant` is the default; every method returns
  "unavailable", so the planner always uses its deterministic fallback. Nothing
  about the deterministic pipeline depends on the LLM sidecar being present.
- Opt-in, local-only. The only implementation shipped alongside
  `DisabledAssistant` is `persona_engine::llm::LmStudioAssistant`, which talks to
  a LOCAL LM Studio server (an OpenAI-compatible `POST /v1/chat/completions`
  endpoint on loopback, e.g. `127.0.0.1:1234`) over plain HTTP. There is no cloud
  provider, no bundled model, and no config path that reaches a non-loopback host
  by default; enabling it is an explicit `--llm` opt-in (`LlmConfig { enabled:
  true, .. }` at the API level).
- Schema-validated output. Anything the sidecar returns is length/scope checked
  and then re-gated by the harmful-query blocklist and the Safety Gate. A
  proposed sub-interest must also name a category the persona is actually
  allowed to have (`propose_subseed`), and still has to clear the appraisal
  Venn gate before it can be noticed, let alone adopted.
- Bounded, multiplicative influence only. `appraise_salience` returns a rating
  in `[0, 1]` that is folded into the deterministic salience as a `[0.5, 1.5]`
  multiplier (`appraise::appraise`); it can amplify or damp what gets noticed,
  but the hard category Venn / safety gate is evaluated first and the LLM is
  never consulted for anything the deterministic path has already ruled out.
- Deterministic fallback required, fail-closed. A disabled config, a connection
  error, a timeout, or a malformed/out-of-range response are all treated
  identically to "unavailable": the planner falls back to the deterministic
  path exactly as if the sidecar had never been consulted. The sync bridge from
  the (synchronous) `SemanticAssistant` trait into the async transport
  (`tokio::task::block_in_place` + `Handle::current().block_on`) requires being
  called from a multi-threaded Tokio runtime, which `fauxx-cli`'s
  `#[tokio::main]` provides; this is a documented constraint of an advanced,
  opt-in feature, never exercised unless `--llm` is passed.

## Dry-run mode

`persona-engine plan` (and `run-once --dry-run`) run the whole pipeline and print
the decision without touching the network or writing to the store. The report
shows the persona id, the current routine, the behavior state (energy, curiosity,
boredom, topic scores), the goal/utility scores, the selected goal, the action
type, the category and subcategory, the candidate intents, whether the sidecar
was used, the safety decision per intent, the final approved plan, and an
explicit `no_network: true` line.

## Safety gates

Two layers, both fail-closed:

- The isolated-profile guard refuses to launch a decoy profile that is not
  verifiably separate from a real browser profile.
- The Safety Gate refuses any intent that: uses a module the policy does not
  allow; requests a forbidden capability (login, create account, submit form,
  comment, post, upload, message, review, purchase, checkout, book, apply, vote,
  contact, click ads, bypass captcha); targets a forbidden or non-allowed
  category; carries an empty or harmful-blocklisted query; or names a domain hint
  that fails the auth-flow / HTTPS navigation guard. Rejections are recorded as
  skips with a reason, logged locally, and never dispatched.

## CLI

```sh
fauxx-cli persona-engine list
fauxx-cli persona-engine show elias_rickensworth
fauxx-cli persona-engine validate elias_rickensworth        # or a path to a .toml
fauxx-cli persona-engine plan --persona elias_rickensworth --dry-run
fauxx-cli persona-engine run-once --persona elias_rickensworth --dry-run
fauxx-cli persona-engine run-once --persona elias_rickensworth   # drives Chromium
fauxx-cli persona-engine logs export --persona elias_rickensworth --format jsonl

# Optional: augment appraisal/discovery with a local LM Studio model. Off by
# default; requires an LM Studio server already running and listening locally.
fauxx-cli persona-engine plan --persona elias_rickensworth --dry-run \
    --llm --llm-endpoint 127.0.0.1:1234 --llm-model local-model
```

`--seed` makes a pass reproducible and `--now <epoch-millis>` overrides the clock,
so a run is fully deterministic for tests and scripted schedules. `--llm` is
available on both `plan` and `run-once`; `--llm-endpoint`/`--llm-model` are
ignored unless `--llm` is also passed.

## Example dry-run output

```
persona_id:      elias_rickensworth
policy_version:  v1
routine:         weekday_evening
state:           energy=0.85 curiosity=0.95 boredom=0.00
goal_scores:     CRAFTS=0.57 HISTORY=0.57 OUTDOOR_RECREATION=0.57
selected_goal:   continue_hobby_thread (elias_rickensworth:weekday_evening:CRAFTS:20)
action_type:     search
category:        CRAFTS / blue black ink
reason:          Elias Rickensworth is in the weekday_evening routine and continues a CRAFTS thread about blue black ink.
sidecar_used:    false
candidate_intents:
  - [CRAFTS] blue black ink
  - [CRAFTS] fountain pen ink for everyday notebook paper
  ...
safety:
  - ALLOW elias_rickensworth:weekday_evening:CRAFTS:20#0: passed all safety checks
  ...
final_plan:
  - [CRAFTS] "blue black ink"  dwell=25-90s  visit<=1
  - [CRAFTS] "fountain pen ink for everyday notebook paper"  dwell=25-90s  visit<=1
no_network:      true
```

## Persistence

Behavior state (one row per policy id) and the activity log (append-only) live in
the encrypted store (schema v15 -> v16). The activity records are decoy-only
synthetic data: routine, goal, category, the synthetic query, the search engine
domain, dwell, and the safety outcome. They carry no secrets, tokens, cookies,
real URLs, or real-user identifiers (a log-schema test freezes the field set).

## Limitation

This system does not defeat logged-in account tracking, payment identity,
phone-number identity, deterministic first-party telemetry, or legal or
process-based identification. It only attempts to dilute weak probabilistic
behavioral profiling by adding bounded decoy signals. The safety model is not
optional; Elias is.

## Out of scope in the MVP (follow-ups)

Ad clicking, mock location, form or account or social interaction, arbitrary
open-ended browsing, and full GOAP/HTN planning (the desire -> goal -> action
selection in the world-model is deliberately GOAP-adjacent, not full GOAP). No
cloud LLM call of any kind is ever made; the only network path the LLM sidecar
can take is loopback to an operator-run LM Studio instance, and even that is
off unless explicitly enabled (see "LLM sidecar limits" above). Extracting real
candidate topics from a LIVE visited page's rendered text (`classify_page_text`
against actual DOM content, rather than the dry-simulation's synthetic
`from_dispatched_query` stimulus) remains unimplemented.
