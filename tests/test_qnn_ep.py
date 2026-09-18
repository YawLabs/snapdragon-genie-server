"""Tests for the ONNX/QNN placement-verification helpers.

This module is the honesty mechanism of the ONNX path: without it a silent CPU
fallback reports itself as an NPU result at roughly a tenth of the speed and
nobody notices, because `get_providers()` still says QNNExecutionProvider. So
the checks that decide "did this really run on the HTP" have to be right, and
one of them could previously return a verdict it had no way to reach.

`onnxruntime` and `onnxruntime_qnn` are not installed here (they are a
Snapdragon-only dependency), so they are stubbed at import time. That stubbing
covers the module's IMPORTS, not the code under test: every function exercised
below runs for real. Where a test needs an InferenceSession, the stand-in does
one thing -- it writes known bytes to fd 1 during construction, the way the
real QNN logger does at the C level -- so the fd capture and the marker
detection run for real on those bytes. What a real driver PRINTS is the part a
stub would happily get wrong, and that stays in a hardware-gated test; nothing
here asserts that the strings in HTP_COMPILE_MARKERS are what QAIRT emits.
"""

import importlib
import os
import sys
import types

import pytest


class _SessionOptions:
    """Enough of ort.SessionOptions for build_session to configure: the
    severity attribute it sets and the attach call it makes."""

    def __init__(self):
        self.log_severity_level = None
        self.attached = []

    def add_provider_for_devices(self, devices, opts):
        self.attached.append((list(devices), dict(opts)))


def _ort_stub():
    ort = types.ModuleType("onnxruntime")
    ort.SessionOptions = _SessionOptions
    # Construction fails by default: a test that wants a session install one
    # with _session_writing, so nothing passes a build it did not ask for.
    ort.InferenceSession = object
    ort.OrtHardwareDeviceType = types.SimpleNamespace(NPU="NPU", GPU="GPU")
    ort.get_ep_devices = lambda: []
    ort.register_execution_provider_library = lambda *a: None
    return ort


@pytest.fixture
def qe(monkeypatch):
    """qnn_ep with its Snapdragon-only imports stubbed."""
    qnn = types.ModuleType("onnxruntime_qnn")
    qnn.get_ep_name = lambda: "QNNExecutionProvider"
    qnn.get_library_path = lambda: "stub.dll"
    monkeypatch.setitem(sys.modules, "onnxruntime", _ort_stub())
    monkeypatch.setitem(sys.modules, "onnxruntime_qnn", qnn)
    import qnn_ep
    return importlib.reload(qnn_ep)


def _npu_device():
    return types.SimpleNamespace(ep_name="QNNExecutionProvider",
                                 device=types.SimpleNamespace(type="NPU"))


def _gpu_device():
    return types.SimpleNamespace(ep_name="QNNExecutionProvider",
                                 device=types.SimpleNamespace(type="GPU"))


def _session_writing(native, fail=None):
    """An InferenceSession stand-in whose construction writes `native` at the
    OS level (fd 1, where ORT's C-side logger goes) and, if `fail` is given,
    raises it afterwards -- a build that printed a diagnostic and then died."""
    class Session:
        def __init__(self, path, sess_options=None):
            self.path = path
            self.sess_options = sess_options
            self.fallback_disabled = False
            os.write(1, native)
            if fail is not None:
                raise fail

        def disable_fallback(self):
            self.fallback_disabled = True

        def get_providers(self):
            return ["QNNExecutionProvider", "CPUExecutionProvider"]
    return Session


def _ident(fd):
    """What fd points at: (device, inode) from the OS, not from Python."""
    st = os.fstat(fd)
    return (st.st_dev, st.st_ino)


def _open_fds(limit=512):
    """How many descriptors below `limit` are open, asked of the OS.

    A count, not "the lowest free fd": the capture's temp file always takes
    the lowest slot and is closed on exit, so a leaked saved descriptor sits
    ABOVE it and the lowest free number never moves. That probe passed with
    the closes deleted; this one climbs by two per call.
    """
    n = 0
    for fd in range(limit):
        try:
            os.fstat(fd)
        except OSError:
            continue
        n += 1
    return n


# --- the imports --------------------------------------------------------------

def test_a_missing_wheel_is_named_at_import(monkeypatch):
    # The old hint ("Is onnxruntime-qnn installed...") lived in a function that
    # cannot run without the package, so the case it named never reached it.
    # The import is where that case actually fails, and it points at the fix.
    monkeypatch.setitem(sys.modules, "onnxruntime", _ort_stub())
    monkeypatch.setitem(sys.modules, "onnxruntime_qnn", None)   # import -> ImportError
    monkeypatch.delitem(sys.modules, "qnn_ep", raising=False)
    with pytest.raises(ImportError) as e:
        importlib.import_module("qnn_ep")
    assert "requirements.txt" in str(e.value)
    assert "onnxruntime_qnn" in str(e.value.__cause__)


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
    # with any severity. It must fail LATER (here: no NPU device in the stub),
    # not at the guard.
    with pytest.raises(Exception) as e:
        qe.build_session("m.onnx", log_severity=4, verify=False)
    assert not isinstance(e.value, ValueError) or "cannot be verified" not in str(e.value)


def test_the_cpu_baseline_is_never_blocked_by_the_guard(qe):
    # use_npu=False makes no placement claim at all; the guard must not fire.
    with pytest.raises(Exception) as e:
        qe.build_session("m.onnx", use_npu=False, log_severity=4)
    assert "cannot be verified" not in str(e.value)


# --- picking the device --------------------------------------------------

def test_no_device_names_the_kind_and_lists_what_exists(qe):
    # The stub's empty device list is exactly a machine without an HTP; this
    # branch runs for real with no hardware and is the message a user on the
    # wrong box reads. It must not ask whether the package is installed --
    # it cannot run unless it is.
    with pytest.raises(RuntimeError) as e:
        qe.get_qnn_device("NPU")
    msg = str(e.value)
    assert "No QNN NPU device" in msg
    assert "[]" in msg, "must list what IS there, even when that is nothing"
    assert "Is onnxruntime-qnn installed" not in msg


def test_the_gpu_kind_is_reported_by_name_too(qe):
    with pytest.raises(RuntimeError) as e:
        qe.get_qnn_device("GPU")
    assert "No QNN GPU device" in str(e.value)


def test_the_requested_kind_decides_which_of_the_two_devices_comes_back(qe):
    # The module docstring's second key fact: get_ep_devices() returns BOTH a
    # QNN NPU and a QNN GPU device. Nothing promises which comes first, so the
    # Adreno leads here -- "take whatever QNN handed back" would answer an NPU
    # request with the GPU and the bench row would still be labelled NPU.
    gpu, npu = _gpu_device(), _npu_device()
    qe.ort.get_ep_devices = lambda: [gpu, npu]
    assert qe.get_qnn_device("NPU") is npu
    assert qe.get_qnn_device("GPU") is gpu


def test_a_qnn_gpu_is_never_handed_back_as_the_npu(qe):
    # QNN reporting only the Adreno: the NPU request has to fail rather than
    # settle for the device that is there, and the message lists what WAS
    # found so the reader can see the GPU was seen and skipped.
    qe.ort.get_ep_devices = lambda: [_gpu_device()]
    with pytest.raises(RuntimeError) as e:
        qe.get_qnn_device("NPU")
    msg = str(e.value)
    assert "No QNN NPU device" in msg
    assert "GPU" in msg, "must name the device it refused to substitute"
    assert "[]" not in msg


def test_a_session_asked_for_the_npu_attaches_the_npu_not_the_first_qnn_device(qe):
    # End of the same thread: the kind filter is the only device-free thing
    # standing between "labelled NPU" and "measured on the Adreno", so pin it
    # where it lands -- in the attach that decides where the graph compiles.
    gpu, npu = _gpu_device(), _npu_device()
    qe.ort.get_ep_devices = lambda: [gpu, npu]
    qe.ort.InferenceSession = _session_writing(b"VTCM Allocation\n")
    session, _ = qe.build_session("m.onnx")
    assert session.sess_options.attached == [
        ([npu], {"htp_performance_mode": "burst"})]


# --- the verdict itself ---------------------------------------------------
# Driven through build_session with a session whose construction writes to
# fd 1, so the capture and the marker check are the real ones. The old test
# here re-implemented the `any(m in log ...)` expression inline and asserted
# a tautology that build_session could have dropped the check under.

def test_a_marker_printed_during_the_build_verifies_the_session(qe):
    qe.ort.get_ep_devices = lambda: [_npu_device()]
    for marker in qe.HTP_COMPILE_MARKERS:
        qe.ort.InferenceSession = _session_writing(
            b"...unrelated...\n" + marker.encode() + b": 3 ops\n...more...\n")
        session, info = qe.build_session("m.onnx")
        assert info["htp_verified"], marker
        assert marker in info["log"]
        assert session.fallback_disabled, "a real NPU run must not retry on CPU"


def test_a_silent_build_is_a_placement_error_when_verified(qe):
    qe.ort.get_ep_devices = lambda: [_npu_device()]
    qe.ort.InferenceSession = _session_writing(b"Session created.\n")
    with pytest.raises(qe.PlacementError) as e:
        qe.build_session("m.onnx")
    # The refusal carries what WAS printed, so the reader can see it was not
    # a compile stage rather than take the verdict on faith.
    assert "Session created." in str(e.value)


def test_the_same_silent_build_is_returned_unverified_when_no_claim_is_made(qe):
    qe.ort.get_ep_devices = lambda: [_npu_device()]
    qe.ort.InferenceSession = _session_writing(b"Session created.\n")
    session, info = qe.build_session("m.onnx", verify=False)
    assert info["htp_verified"] is False
    assert "Session created." in info["log"]


def test_the_npu_is_attached_through_the_plugin_path_with_the_perf_mode(qe):
    # The legacy providers=[...] argument is silently ignored under the
    # dynamic-EP model (module docstring); add_provider_for_devices is the
    # only attach that places anything, and it must carry the options.
    device = _npu_device()
    qe.ort.get_ep_devices = lambda: [device]
    qe.ort.InferenceSession = _session_writing(b"VTCM Allocation\n")
    session, _ = qe.build_session("m.onnx", perf_mode="high_performance",
                                  extra_options={"x": "1"}, log_severity=2)
    assert session.sess_options.attached == [
        ([device], {"htp_performance_mode": "high_performance", "x": "1"})]
    assert session.sess_options.log_severity_level == 2


def test_the_cpu_baseline_attaches_nothing(qe):
    qe.ort.InferenceSession = _session_writing(b"")
    session, info = qe.build_session("m.onnx", use_npu=False, verify=False)
    assert session.sess_options.attached == []
    assert info["htp_verified"] is False
    assert not session.fallback_disabled, "the CPU session keeps ORT's default"


def test_a_failed_build_keeps_the_native_diagnostic(qe):
    # fd 2 is redirected for the whole build, so the HTP backend's own error
    # text lands in the capture and ORT's exception does not carry it. It
    # used to be filled into box["text"] by the finally and read by no one.
    qe.ort.get_ep_devices = lambda: [_npu_device()]
    qe.ort.InferenceSession = _session_writing(
        b"QnnHtp: backend init failed, error 1003\n", fail=RuntimeError("ORT: generic failure"))
    with pytest.raises(RuntimeError) as e:
        qe.build_session("m.onnx")
    assert str(e.value) == "ORT: generic failure", "same exception, not a rewrap"
    assert "error 1003" in "".join(getattr(e.value, "__notes__", []))


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


# --- fd bookkeeping -------------------------------------------------------
# Run for real against the OS. A leak here silently redirects the rest of the
# process's stdout into a closed temp file: every later print vanishes and it
# presents as a hang rather than an error. The checks compare what the OS says
# a descriptor points at, because a write to a wrongly-redirected fd 1 still
# SUCCEEDS (the temp file is open) -- the previous version of these tests did
# exactly that write and passed with the restore deleted.

def test_capture_restores_the_descriptors_when_the_body_raises(qe):
    before_out, before_err = _ident(1), _ident(2)
    with pytest.raises(RuntimeError):
        with qe.capture_native_output():
            raise RuntimeError("session build failed")   # the NORMAL path
    assert _ident(1) == before_out
    assert _ident(2) == before_err   # also catches a swapped restore


def test_capture_collects_native_output(qe):
    with qe.capture_native_output() as box:
        os.write(1, b"Finalizing Graph Sequence\n")
    assert "Finalizing Graph Sequence" in box["text"]


def test_capture_collects_stderr_too(qe):
    # The QNN backend's own errors go to fd 2; both must land in one buffer.
    with qe.capture_native_output() as box:
        os.write(2, b"QnnHtp error\n")
    assert "QnnHtp error" in box["text"]


def test_capture_fills_the_box_even_when_the_body_raises(qe):
    box = None
    with pytest.raises(RuntimeError):
        with qe.capture_native_output() as box:
            os.write(2, b"backend said no\n")
            raise RuntimeError("build failed")
    assert "backend said no" in box["text"]


def test_capture_does_not_leak_descriptors_across_repeated_use(qe):
    # Called once per session build; a per-call leak exhausts the fd table on a
    # long sweep rather than failing on the first call. The number of open
    # descriptors is constant while every dup is matched by a close.
    before = _open_fds()
    for _ in range(20):
        with qe.capture_native_output() as box:
            os.write(1, b"x")
        assert box["text"] == "x"
    assert _open_fds() == before


def test_a_redirect_that_fails_half_way_is_undone(qe, monkeypatch):
    # fd 1 is pointed at the temp file before fd 2 is. If the second dup2
    # raises, the first must still be reverted and the saved copies closed;
    # a setup outside the try left fd 1 in the temp file for good.
    real_dup2 = os.dup2
    armed = {"v": True}

    def flaky_dup2(fd, fd2, *rest):
        if fd2 == 2 and armed["v"]:
            armed["v"] = False
            raise OSError(9, "simulated failure redirecting fd 2")
        return real_dup2(fd, fd2, *rest)
    before_out = os.dup(1)
    before = _open_fds()
    monkeypatch.setattr(os, "dup2", flaky_dup2)
    try:
        with pytest.raises(OSError, match="simulated"):
            with qe.capture_native_output():
                pytest.fail("the body must not run after a failed redirect")
        assert _ident(1) == _ident(before_out)
        assert _open_fds() == before
    finally:
        real_dup2(before_out, 1)   # keep the rest of the run sane if it did leak
        os.close(before_out)
