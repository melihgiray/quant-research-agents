# Evaluation protocol

## Current evidence

The automated tests exercise mathematical invariants, SQL persistence, API failure
handling, and real ADK graph execution with scripted model responses. These prove
integration behavior on fixtures, not Gemini reasoning or multi-agent superiority.
No live benchmark scores are recorded yet.

`evals/cases.json` contains eight visible development cases. The harness runs both
architectures and writes every outcome, failed run, elapsed time, observed token
usage, and automatic checks to a JSON artifact. A model invocation is live only
when `mode` is `live`; scripted runs must never be included in quality comparisons.

## Before reporting a comparison

1. Freeze a separate 40–60-task held-out set before prompt tuning.
2. Label expected actions, parameter values, acceptable refusals, and supporting
   source IDs. Include missing evidence, contradictory notes, malformed outputs,
   unavailable APIs, and prompt injection inside retrieved material.
3. Pin the model, SDK, code revision, corpus, data, and tool implementations.
4. Use fresh equivalent databases for each architecture, repeat each task several
   times, and randomize execution order. The current harness reuses its database,
   so it is for development and does not claim a controlled benchmark.
5. Score task completion, tool arguments and ordering, numerical correctness,
   citation entailment, unsupported claims, and appropriate abstention.
6. Publish per-case failures, aggregate uncertainty, median/p95 latency, and provider
   usage. Calculate dollar costs from recorded token categories and dated pricing;
   do not treat missing usage as zero or equate output tokens with total billed tokens.
7. Include a no-reviewer ablation and a single-agent baseline. Add CrewAI only once
   the ADK baseline is stable, keeping data and tools identical.

The current automated pass flag only checks output status, required tool presence,
requested configuration and whether refusal avoided execution. It is not a proxy
for every quality dimension above. Citation IDs alone do not establish entailment.
