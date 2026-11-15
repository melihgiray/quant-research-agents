from quant_research_agents.evaluation import score_case


def test_refusal_after_execution_is_not_evaluation_success():
    score = score_case(
        {"expected_status": "refused"}, {"status": "refused", "tool_calls": ["run_backtest"]}
    )
    assert score["automated_pass"] is False


def test_completed_status_without_report_is_not_success():
    score = score_case(
        {"expected_status": "completed", "required_tools": ["run_backtest"]},
        {"status": "completed", "tool_calls": ["run_backtest"]},
    )
    assert score["automated_pass"] is False
