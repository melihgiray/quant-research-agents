# Run and explain the project

## One-command offline walkthrough

After the README installation steps, run from the repository root:

```sh
quant-agents doctor
python scripts/demo_portfolio.py
```

The script prints a path to `walkthrough.json`. Each invocation gets a fresh folder
under `workspace/portfolio-demo/`, with separate databases for the two architectures.
It ignores configured Gemini and PostgreSQL credentials and makes no provider calls.

The walkthrough executes both ADK graphs with scripted model responses, checks their
saved reports against fresh synthetic replays, evaluates the twelve retrieval queries,
and compares the default experiment with a 20-basis-point cost fixture. The additional
cost fixture is executed directly in Python and is explicitly labeled as such.

Open the HTML path in either architecture's `run` object. Inspect `trace.jsonl` beside
the run's `run.json` to follow retrieval, SQL history, and experiment calls. Read the
provenance fields to identify the code, corpus and dependencies behind the result.

## Explain the engineering decisions

1. **Agent collaboration:** `agents.py` constructs the coordinator, researcher,
   analyst and reviewer graph. Typed plans and canonical session state carry the
   contract between roles. A single-agent baseline uses the same tools.
2. **Tool execution:** `tools.py` enforces retrieval and SQL-history prerequisites,
   parameter bounds, and agreement with the coordinator's plan. Calculations run
   in Python; the model cannot supply SQL, Python code or replacement price data.
3. **Retrieval:** `retrieval.py` implements BM25 over five attributed notes.
   `evidence.py` validates IDs and references and fingerprints content. These checks
   do not turn document text into trusted instructions or prove citation entailment.
4. **Verification:** the finalizer checks typed claims before saving. An optional
   correction agent gets one attempt without execution tools. `verification.py`
   can independently compare a saved artifact to a fresh synthetic replay.
5. **Reproducibility:** SQL writes are idempotent; each invocation has its own trace
   and report. Code/corpus fingerprints and package versions travel with run metadata.
6. **Evaluation:** single/multi runs use fresh databases and shuffled repeated trials.
   Summaries retain failures and missing usage. The visible cases are development
   fixtures, not a held-out benchmark.

## Commands to explore

```sh
quant-agents evaluate-retrieval --k 3 --output workspace/retrieval.json
quant-agents verify PATH_TO_REPORT_JSON
quant-agents --workspace PATH_TO_RUN_WORKSPACE history
quant-agents --workspace PATH_TO_RUN_WORKSPACE compare LEFT_ID RIGHT_ID
pytest -q
```

The verification command checks known evidence content and synthetic replay; it
cannot establish who authored a report or whether its prose is factually supported.
Comparison refuses mismatched datasets, engine versions and scoring windows.

## Move from scripted execution to model evidence

Set your Gemini key locally, select a concrete available model ID in `QUANT_MODEL`,
and run a supported prompt with `quant-agents run`. The scripted walkthrough is not
a substitute for this step. Preserve the live report and trace, inspect parameter
choices, and check claims and citation support yourself.

```sh
quant-agents evaluate --live --trials 3 --seed 42
quant-agents summarize PATH_TO_EVALUATION_JSON
quant-agents evaluate --live --trials 3 --seed 42 --no-review-repair
```

Each evaluation command above schedules 48 live runs for the eight supplied cases
and can incur provider charges. Keep model and execution limits identical for the
correction comparison. Disabling correction leaves the initial reviewer in place.
Do not claim a measured improvement until repeated live results support it.

## What the portfolio currently demonstrates

Implemented and fixture-tested: Python, Google ADK orchestration, function calling,
structured outputs, BM25 retrieval, parameterized SQL, API failure handling, execution
bounds, report verification and an evaluation harness. SQLite and the scripted ADK
paths work locally. The optional PostgreSQL integration needs a test server; live
Gemini and FRED need credentials. CrewAI and the trained hybrid retriever remain
planned. Synthetic strategy metrics are not investment research results.
