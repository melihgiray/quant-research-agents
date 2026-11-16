# Evaluation protocol

## Retrieval-only evaluation

Run `quant-agents evaluate-retrieval --k 3 --output workspace/retrieval.json`
without an API key. Twelve visible cases label relevant document IDs or an empty
set for abstention. Hit rate, mean recall and reciprocal rank use answerable cases
only; abstention accuracy uses unanswerable cases only. An absent group produces
`null`, not a perfect score. The result retains each ranked list and fingerprints
the cases and corpus. This small development set is not a general RAG benchmark.

## Current evidence

The automated tests exercise mathematical invariants, SQL persistence, API failure
handling, and real ADK graph execution with scripted model responses. These prove
integration behavior on fixtures, not Gemini reasoning or multi-agent superiority.
No live benchmark scores are recorded yet.

`evals/cases.json` contains eight visible development cases. The harness runs both
architectures and writes every outcome, failed run, elapsed time, observed token
usage, and automatic checks to a JSON artifact. A model invocation is live only
when `mode` is `live`; scripted runs must never be included in quality comparisons.
Use `--trials 3 --seed 42` to repeat each case three times in a reproducibly shuffled
order. Every attempt uses its own SQLite database regardless of `DATABASE_URL`.
The artifact includes a snapshot and fingerprint of the cases, package versions,
trial numbers, execution order, and reviewer correction counts. Results are saved
after each attempt, so interrupted evaluations retain completed outcomes.

## Before reporting a comparison

`quant-agents summarize path/to/evaluation.json` reports per-architecture pass
counts, failure statuses, median/p95 recorded latency, usage coverage, and correction
frequency. Failed attempts remain in the pass-rate denominator. Missing tokens and
costs stay unknown; sums explicitly cover observed records only. The command refuses
scripted results presented as live evaluations. An interrupted artifact stays labeled
interrupted, and the summary does not imply statistical significance.

1. Freeze a separate 40–60-task held-out set before prompt tuning.
2. Label expected actions, parameter values, acceptable refusals, and supporting
   source IDs. Include missing evidence, contradictory notes, malformed outputs,
   unavailable APIs, and prompt injection inside retrieved material.
3. Pin the model, SDK, code revision, corpus, data, and tool implementations.
4. Use the harness's fresh databases, repeated trials, and randomized execution
   order. Hold model configuration and external context constant between runs;
   API drift and model nondeterminism remain possible confounders.
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
Scripted fault-injection tests cover incorrect numbers and invented citations,
successful correction, exhaustion after one correction, and no re-execution during
correction. These establish control flow, not real-model correction success rates.
