"""Tests for the ONNX/QNN placement-verification helpers.

This module is the honesty mechanism of the ONNX path: without it a silent CPU
fallback reports itself as an NPU result at roughly a tenth of the speed and
nobody notices, because `get_providers()` still says QNNExecutionProvider. So
the checks that decide "did this really run on the HTP" have to be right, and
one of them could previously return a verdict it had no way to reach.

`onnxruntime` and `onnxruntime_qnn` are not installed here (they are a
Snapdragon-only dependency), so they are stubbed at import time. That stubbing
covers the module's IMPORTS, not the code under test: every function exercised
below runs for real. Anything that genuinely needs a device -- building a
session, reading a compile log out of ORT -- is deliberately NOT faked, because
a mock of the QNN boundary would happily agree with a wrong expectation. Those
belong in a hardware-gated test.
"""

import importlib
import os
import sys
import types

import pytest

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")


@pytest.fixture
def qe(monkeypatch):
    """qnn_ep with its Snapdragon-only imports stubbed."""
    if SRC not in sys.path:
        sys.path.insert(0, SRC)
    ort = types.ModuleType("onnxruntime")
    ort.SessionOptions = object
    ort.InferenceSession = object
    ort.OrtHardwareDeviceType = types.SimpleNamespace(NPU="NPU", GPU="GPU")
    ort.get_ep_devices = lambda: []
    ort.register_execution_provider_library = lambda *a: None
    qnn = types.ModuleType("onnxruntime_qnn")
    qnn.get_ep_name = lambda: "QNNExecutionProvider"
    qnn.get_library_path = lambda: "stub.dll"
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)
    monkeypatch.setitem(sys.modules, "onnxruntime_qnn", qnn)
    import qnn_ep
    return importlib.reload(qnn_ep)


# --- the severity trap ----------------------------------------------------
# The markers only exist in the log at severity <= 3. A quieter session cannot
# emit them even on a perfectly placed graph, so verification would fail for a
# reason that has nothing to do with placement.

def test_unverifiable_severity_is_refused_not_guessed(qe):
    with pytest.raises(ValueError) as e:
        qe.build_session("m.onnx", log_severity=4)
    msg = str(e.value)
    assert "cannot be verified" in msg
    assert "<= 3" in msg, "must say what value would work"


def test_the_refusal_names_the_deliberate_escape_hatch(qe):
    # Quieting the log is legitimate -- it just cannot be combined with a
    # placement claim, and the caller needs to be told how to do it honestly.
    with pytest.raises(ValueError) as e:
        qe.build_session("m.onnx", log_severity=9)
    assert "verify=False" in str(e.value)


def test_a_quiet_session_is_allowed_when_no_claim_is_made(qe):
    # verify=False means "I am not asserting placement", which is compatible
    # with any severity. It must fail LATER (at the stubbed ORT call), not at
    # the guard.
    with pytest.raises(Exception) as e:
        qe.build_session("m.onnx", log_severity=4, verify=False)
    assert not isinstance(e.value, ValueError) or "cannot be verified" not in str(e.value)


def test_the_cpu_baseline_is_never_blocked_by_the_guard(qe):
    # use_npu=False makes no placement claim at all; the guard must not fire.
    with pytest.raises(Exception) as e:
        qe.build_session("m.onnx", use_npu=False, log_severity=4)
    assert "cannot be verified" not in str(e.value)


# --- the verdict itself ---------------------------------------------------

def test_placement_assertion_rejects_an_unverified_session(qe):
    with pytest.raises(qe.PlacementError):
        qe.assert_htp_placement({"htp_verified": False, "providers": ["CPUExecutionProvider"]})


def test_placement_assertion_passes_a_verified_session(qe):
    qe.assert_htp_placement({"htp_verified": True, "providers": ["QNNExecutionProvider"]})


def test_a_missing_key_is_treated_as_unverified(qe):
    # Absence of evidence must not read as evidence: a malformed info dict is
    # the one case where defaulting to "verified" would be silently wrong.
    with pytest.raises(qe.PlacementError):
        qe.assert_htp_placement({})


def test_every_marker_counts_as_proof(qe):
    # Which stage a given driver prints varies; any one of them is proof the
    # graph compiled for the Hexagon target.
    for marker in qe.HTP_COMPILE_MARKERS:
        log = "...unrelated...\n%s: 3 ops\n...more...\n" % marker
        assert any(m in log for m in qe.HTP_COMPILE_MARKERS), marker


# --- fd bookkeeping -------------------------------------------------------
# Run for real against the OS. A leak here silently redirects the rest of the
# process's stdout into a closed temp file: every later print vanishes and it
# presents as a hang rather than an error.

def test_capture_restores_the_descriptors_when_the_body_raises(qe):
    before_out, before_err = os.dup(1), os.dup(2)
    try:
        with pytest.raises(RuntimeError):
            with qe.capture_native_output():
                raise RuntimeError("session build failed")   # the NORMAL path
        # If fd 1 were still pointing at the temp file, this write would go
        # nowhere; that it is writable and the process still functions is the
        # property under test.
        os.write(1, b"")
        after = os.dup(1)
        os.close(after)
    finally:
        os.close(before_out)
        os.close(before_err)


def test_capture_collects_native_output(qe):
    with qe.capture_native_output() as box:
        os.write(1, b"Finalizing Graph Sequence\n")
    assert "Finalizing Graph Sequence" in box["text"]


def test_capture_does_not_leak_descriptors_across_repeated_use(qe):
    # Called once per session build; a per-call leak exhausts the fd table on a
    # long sweep rather than failing on the first call.
    for _ in range(20):
        with qe.capture_native_output() as box:
            os.write(1, b"x")
        assert box["text"] == "x"
