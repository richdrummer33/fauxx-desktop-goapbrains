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
Persona Policy -> Behavior Kernel -> Goal Layer -> Planner
    -> (optional) LLM Sidecar -> Safety Gate -> Executor -> Logs
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
- Goal Layer (`persona_engine::goal`): the control plane. It scores the routine's
  candidate categories with the utility model (see below) and draws ONE with a
  temperature softmax, then fills in a goal: routine, goal type, action type,
  category, subcategory, budget, allowed modules, and a boring human reason.
- Planner (`persona_engine::planner`): turns the goal into structured candidate
  intents carrying the actual search query. Rules-first and deterministic; it
  never emits an arbitrary URL, and it keeps the persona IN CHARACTER (see below).
- LLM Sidecar (`persona_engine::sidecar`): an optional helper. See limits below.
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
page text, or writes a boring reason for an already-chosen action.

Rules enforced in code:

- Disabled by default. The MVP ships only `DisabledAssistant`; there is no cloud
  provider and no local model. Every method returns "unavailable", so the planner
  always uses its deterministic fallback.
- Schema-validated output. Anything the sidecar returns is length/scope checked
  and then re-gated by the harmful-query blocklist and the Safety Gate.
- Deterministic fallback required. A disabled, erroring, or invalid sidecar is
  indistinguishable to the planner from one that was never there.

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
```

`--seed` makes a pass reproducible and `--now <epoch-millis>` overrides the clock,
so a run is fully deterministic for tests and scripted schedules.

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
open-ended browsing, any cloud or local LLM model call, and GOAP/HTN
micro-planning. The MVP pipeline stays Routine -> Goal -> ActionType -> Category
-> Args -> Safety -> Execute.
