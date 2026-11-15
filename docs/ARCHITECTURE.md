# Architecture

The first release answers one bounded request using real tools and either Gemini
or a clearly labeled scripted integration fixture. Both orchestration modes share
the tools, data, report schemas, and deterministic checks.

## Boundaries

1. `models.py`: Pydantic contracts and parameter bounds.
2. `agents.py`: ADK 2 `Workflow`, role instructions and finalization.
3. `tools.py`: model-facing operations, prerequisite checks, and session state.
4. `experiment.py`: deterministic calculations and provenance.
5. `store.py`: parameterized SQL, portable SQLite/PostgreSQL persistence.
6. `retrieval.py`: BM25 evidence retrieval over attributed local notes.
7. `reporting.py`: numerical verification and inert HTML/Markdown/JSON exports.
8. `runtime.py`: isolated sessions, bounds, traces, and provider usage metadata.

## Why these choices

- **Explicit graph order:** the analyst needs retrieved evidence; the reviewer
  needs completed results. Parallel retrieval and history lookup can be tested
  later if latency measurements justify another workflow node.
- **Tools calculate; models interpret:** metrics are derived from saved return
  streams. The model cannot supply executable Python, filesystem paths, SQL, URLs,
  or a replacement dataset to the execution tool.
- **State carries authoritative evidence:** sibling agents’ tool messages are not
  assumed to be forwarded. The reviewer receives canonical result and evidence
  objects from session state, not just an analyst’s prose.
- **Separate single-agent baseline:** identical functions and final checks make
  orchestration a measurable design choice, rather than an assumed improvement.
- **Small local corpus:** makes clone-and-run simple while preserving a real
  retrieval boundary for the existing hybrid retriever later.
- **SQL idempotency:** experiment identity hashes configuration, engine version,
  and dataset fingerprint. Identical executions share an experiment record;
  individual sessions have separate report directories and traces.

## Failure semantics

Invalid tool arguments return typed status messages so an agent can correct them
within its call budget. FRED has two bounded attempts. Unsupported requests should
produce explicit refusal; model adherence is a live evaluation target. Missing
execution results, invented citation IDs, and wrong structured numbers fail report
finalization. They cannot produce a verified report. The report's free-form prose
is not covered by numerical validation.

CLI conversations use `InMemorySessionService`; persistent experiment records do
not imply resumable conversational state. The first version does not implement
durable background jobs or deployment authentication.

## Reproduction

`requirements-tested.txt` captures the development environment’s package versions.
Use it as a constraints file with the project install. ADK is explicitly pinned in
`pyproject.toml`. Model aliases can change; select a concrete provider model ID for
benchmarks, record that configuration, and retain run artifacts.
