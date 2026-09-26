from infovore.llm.process import ProcessResult


def test_process_result_reports_timeout_separately_from_exit_code() -> None:
    result = ProcessResult(exit_code=None, stdout="", stderr="", timed_out=True)
    assert result.timed_out
    assert result.exit_code is None
