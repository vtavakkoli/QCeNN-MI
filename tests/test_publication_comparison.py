import pytest

from qcenn_mi.publication_comparison import (
    assert_manuscript_table,
    build_report,
    summarize,
)


def _case():
    resident = {"accuracy_passed": True, "charged_queries_per_snapshot": 8, "trace_error": 0.0}
    fresh = {"accuracy_passed": False, "charged_queries_per_snapshot": 8, "trace_error": 0.0}
    direct = {
        "accuracy_passed": True, "charged_queries_per_snapshot": 8,
        "trace_error": 0.0, "postselection_success_probability": 0.5,
        "expected_queries_one_accepted_preparation": 16,
    }
    return {
        "case": {"snapshots": 1, "drift": "slow", "logical_noise_per_oracle_call": 0.0,
                 "query_budget_per_snapshot": 8},
        "summary": {"resident": {"passed": 1}, "fresh": {"passed": 0},
                    "one_shot": {"passed": 1}},
        "trajectory": [{"observations": {"resident": resident, "fresh": fresh,
                                         "one_shot": direct}}],
        "independent_check": {"status": "passed"},
    }


def test_fresh_resident_summary_is_paired_and_preserves_postselection_cost():
    result = summarize([_case()])
    assert result["number_of_cases"] == 1
    assert result["passing"] == {"resident": 1, "fresh": 0, "one_shot": 1}
    assert result["paired"]["resident_wins"] == 1
    assert result["one_shot_expected_calls_over_single_attempt_budget"]["median"] == 2.0


def test_temporal_evidence_rejects_over_budget_or_forged_summary():
    case = _case()
    case["trajectory"][0]["observations"]["resident"]["charged_queries_per_snapshot"] = 9
    with pytest.raises(ValueError, match="budget"):
        summarize([case])
    case = _case()
    case["summary"]["fresh"]["passed"] = 1
    with pytest.raises(ValueError, match="differs"):
        summarize([case])


def test_paper_table_noisy_math_row_is_parsed_and_mismatch_fails(tmp_path):
    paper = tmp_path / "paper.tex"
    paper.write_text(r"""
All snapshots (576) & 154 & 87 & 194\\
Gradual change (288) & 93 & 42 & 96\\
Abrupt change (288) & 61 & 45 & 98\\
Logical $p=10^{-3}$ (288) & 0 & 0 & 0\\
""")
    summary = {
        "number_of_cases": 96, "snapshots_per_method": 576,
        "passing": {"resident": 154, "fresh": 87, "one_shot": 194},
        "by_drift": {"slow": {"resident": 93, "fresh": 42, "one_shot": 96},
                     "abrupt": {"resident": 61, "fresh": 45, "one_shot": 98}},
        "by_noise": {"0.001": {"resident": 0, "fresh": 0, "one_shot": 0}},
    }
    assert assert_manuscript_table(paper, summary)["status"] == "passed"
    paper.write_text(paper.read_text().replace("154 & 87", "153 & 87"))
    with pytest.raises(ValueError, match="mismatch"):
        assert_manuscript_table(paper, summary)


def test_report_rejects_missing_phase_validation(tmp_path):
    paper = tmp_path / "paper.tex"
    paper.write_text("nothing")
    with pytest.raises(ValueError, match="phase audit"):
        build_report(
            {"static_cases": 1416, "static_accuracy_passed": 770},
            {"cases": [_case()]},
            {"records": [{"numerical_validity_passed": False}]},
            paper,
        )



def test_missing_historical_summary_is_explicitly_marked_not_available(monkeypatch, tmp_path):
    from qcenn_mi import publication_comparison as module

    monkeypatch.setattr(
        module, "assert_manuscript_table",
        lambda manuscript, summary: {"status": "passed", "verified_rows": []},
    )
    report = module.build_report(
        None,
        {"cases": [_case()]},
        {"records": [{"condition": 2, "degree": 4,
                      "numerical_validity_passed": True,
                      "maximum_qsp_block_error": 0.0,
                      "coefficient_l1_error_estimate": 0.0,
                      "noise_induced_infidelity_bias": 0.0}]},
        tmp_path / "manuscript.tex",
    )
    assert report["previous_static"]["status"] == "not_available"
    assert report["previous_static"]["static_cases"] is None
    assert report["matched_study_verified"] is True


def test_missing_history_does_not_override_temporal_or_phase_failures(tmp_path):
    with pytest.raises(ValueError, match="phase audit"):
        build_report(
            None, {"cases": [_case()]},
            {"records": [{"numerical_validity_passed": False}]},
            tmp_path / "manuscript.tex",
        )


def test_optional_history_cli_skips_absent_default_but_rejects_explicit(
    monkeypatch, tmp_path,
):
    import json
    import sys

    from qcenn_mi import publication_comparison as module

    monkeypatch.chdir(tmp_path)
    manuscript = tmp_path / "manuscript.tex"
    manuscript.write_text("dummy manuscript")
    temporal = tmp_path / "temporal.json"
    temporal.write_text(json.dumps({"cases": [_case()]}))
    phase = tmp_path / "phase.json"
    phase.write_text(json.dumps({"records": [{
        "condition": 2, "degree": 4, "numerical_validity_passed": True,
        "maximum_qsp_block_error": 0.0,
        "coefficient_l1_error_estimate": 0.0,
        "noise_induced_infidelity_bias": 0.0,
    }]}))
    monkeypatch.setattr(
        module, "assert_manuscript_table",
        lambda manuscript, summary: {"status": "passed", "verified_rows": []},
    )
    command = [
        "qcenn_mi.publication_comparison",
        "--temporal", str(temporal), "--phase", str(phase),
        "--manuscript", str(manuscript),
        "--output-dir", str(tmp_path / "comparison"),
    ]
    monkeypatch.setattr(sys, "argv", command)
    module.main()
    report = json.loads((tmp_path / "comparison" / "comparison.json").read_text())
    assert report["previous_static"]["status"] == "not_available"
    with pytest.raises(SystemExit) as failure:
        monkeypatch.setattr(sys, "argv", [
            *command, "--previous", str(tmp_path / "missing-static.json"),
            "--output-dir", str(tmp_path / "unused"),
        ])
        module.main()
    assert failure.value.code == 2
