import hashlib
import json

from qcenn_mi.publication_preflight import inspect, verify_manifest


def test_manifest_verification_rejects_tampering(tmp_path):
    data = tmp_path / "result.json"
    data.write_text('{"passed":true}\n')
    digest = hashlib.sha256(data.read_bytes()).hexdigest()
    (tmp_path / "MANIFEST.json").write_text(json.dumps({
        "sha256": {"result.json": digest}}))
    assert verify_manifest(tmp_path)["status"] == "passed"
    data.write_text('{"passed":false}\n')
    assert verify_manifest(tmp_path)["status"] == "failed"


def test_submission_preflight_never_claims_automatic_editorial_signoff(tmp_path):
    evidence = tmp_path / "run"
    evidence.mkdir()
    (evidence / "result.json").write_text("{}")
    digest = hashlib.sha256((evidence / "result.json").read_bytes()).hexdigest()
    (evidence / "MANIFEST.json").write_text(json.dumps({
        "sha256": {"result.json": digest}}))
    result = inspect(repository=tmp_path, evidence=evidence)
    assert result["evidence_manifest"]["status"] == "passed"
    assert result["automatic_integrity_passed"] is False
    assert result["submission_signed_off"] is False
    assert result["manual_verifications_outstanding"]
