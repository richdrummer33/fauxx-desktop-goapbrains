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
  candidate categories (momentum plus a curiosity-scaled novelty bonus minus a
  cooldown penalty) and chooses ONE goal: routine, goal type, action type,
  category, subcategory, budget, allowed modules, and a boring human reason.
  Occasionally it takes a boring pivot or skips the run entirely so the persona
  is not too coherent (coherence is itself a fingerprint).
- Planner (`persona_engine::planner`): turns the goal into structured candidate
  intents carrying the actual search query. Rules-first and deterministic; it
  never emits an arbitrary URL.
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
