import os
from pathlib import Path

from dotenv import load_dotenv

from quant_research_agents.agents import build_agent

load_dotenv(Path(__file__).resolve().parents[2] / ".env")
root_agent = build_agent(
    Path(os.getenv("QUANT_WORKSPACE", "workspace")), database_url=os.getenv("DATABASE_URL")
)
