# Roadmap

## Milestone 1: executable foundation

- [x] Four Google ADK agents and a deterministic report finalizer.
- [x] Single-agent baseline using the same tools.
- [x] Synthetic experiment fixture with attributed upstream strategy functions.
- [x] Local BM25 evidence, SQL experiment records, optional FRED integration.
- [x] HTML/Markdown/JSON reports, traces and numerical checks.
- [x] Tests and a live development evaluation harness.
- [ ] Verify Gemini end to end with credentials and preserve a live trace.
- [ ] Verify FRED against its live API.

## Milestone 2: research data and retrieval

- [ ] Integrate the existing quant-retrieval hybrid pipeline behind the evidence API.
- [ ] Measure retrieval quality on methodology questions before switching defaults.
- [ ] Add a real dataset import contract with provenance and explicit user selection.
- [ ] Wrap the full quant-research-system engine, including walk-forward splits and costs.
- [ ] Add point-in-time macro data only if it becomes a strategy input.

## Milestone 3: reliability and comparative evidence

- [ ] Freeze a held-out task set and publish repeated live evaluation results.
- [x] Add one bounded reviewer correction with fault-injection integration tests.
- [x] Isolate evaluation databases, shuffle trials, and retain failure artifacts.
- [ ] Measure the correction step's live success rate, benefit, latency and cost.
- [ ] Add a CrewAI implementation of the same workflow and another model provider.
- [ ] Persist conversational sessions and test restart/recovery behavior.
- [ ] Deploy a small authenticated demonstration after local live behavior is verified.
