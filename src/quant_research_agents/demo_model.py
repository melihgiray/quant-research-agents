"""Scripted ADK model for deterministic integration demonstrations, NOT an LLM."""

import json

from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from pydantic import PrivateAttr

from .models import ExperimentConfig, Plan


class ScriptedDemoModel(BaseLlm):
    model: str = "scripted-integration-fixture"
    role: str
    _step: int = PrivateAttr(default=0)

    @property
    def capabilities(self):
        from google.adk.models import LlmCapabilities

        return LlmCapabilities(output_schema_and_tools=True)

    async def generate_content_async(self, llm_request, stream=False):
        config = ExperimentConfig()
        tool_steps = {
            "researcher": [
                ("search_methodology", {"query": "momentum volatility execution costs"})
            ],
            "experiment_analyst": [
                ("list_experiments", {"limit": 5}),
                ("run_backtest", config.model_dump()),
            ],
            "single_analyst": [
                ("search_methodology", {"query": "momentum volatility execution costs"}),
                ("list_experiments", {"limit": 5}),
                ("run_backtest", config.model_dump()),
            ],
        }
        sequence = tool_steps.get(self.role, [])
        if self._step < len(sequence):
            name, args = sequence[self._step]
            self._step += 1
            content = types.Content(
                role="model",
                parts=[types.Part(function_call=types.FunctionCall(name=name, args=args))],
            )
        elif self.role == "coordinator":
            plan = Plan(
                decision="execute",
                reason="Fixed scripted integration demonstration.",
                config=config,
                research_query="momentum volatility execution costs",
            )
            content = types.Content(role="model", parts=[types.Part(text=plan.model_dump_json())])
        elif self.role in {"reviewer", "review_repair", "single_analyst"}:
            responses = {}
            for turn in llm_request.contents:
                for part in turn.parts or []:
                    if part.function_response:
                        responses[part.function_response.name] = part.function_response.response
            if self.role in {"reviewer", "review_repair"}:
                instruction = llm_request.config.system_instruction
                if not isinstance(instruction, str):
                    instruction = "\n".join(p.text or "" for p in instruction.parts)
                fields = {}
                for line in instruction.splitlines():
                    for key in ("EXPERIMENT_JSON", "EVIDENCE_JSON"):
                        if line.startswith(key + ": "):
                            fields[key] = json.loads(line[len(key) + 2 :])
                result = fields["EXPERIMENT_JSON"]
                evidence = fields["EVIDENCE_JSON"]
            else:
                result = responses["run_backtest"]
                evidence = responses["search_methodology"]["documents"]
            review = {
                "status": "completed",
                "summary": "The two variants were executed over the same synthetic dataset and "
                "cost assumptions. The table records their computed outcomes. "
                "This scripted demonstration does not measure model reasoning quality.",
                "citations": [item["id"] for item in evidence],
                "claims": [
                    {"variant": name, "metric": "sharpe", "value": values["sharpe"]}
                    for name, values in result["metrics"].items()
                ],
                "limitations": [
                    "Scripted responses exercise ADK orchestration and real tools; "
                    "no Gemini request was made."
                ],
            }
            content = types.Content(role="model", parts=[types.Part(text=json.dumps(review))])
        else:
            content = types.Content(
                role="model",
                parts=[
                    types.Part(
                        text="Use the exact tool outputs above as evidence. This is synthetic fixture data."
                    )
                ],
            )
        yield LlmResponse(content=content)
