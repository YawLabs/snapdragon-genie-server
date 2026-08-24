"""Reusable helpers for driving the QNN Execution Provider on the Snapdragon
X Elite Hexagon NPU (HTP) via ONNX Runtime's dynamic-EP (plugin) model.

Key facts baked in here (these cost real time to rediscover):

  * onnxruntime-qnn (2.5.0) is a *plugin* for onnxruntime (1.29). You register
    its library, then attach it to a session with `add_provider_for_devices`.

  * The legacy `providers=[("QNNExecutionProvider", {...})]` argument to
    InferenceSession is SILENTLY IGNORED under the dynamic-EP model -- the
    session builds, `get_providers()` can even look plausible, but the op
    runs on CPU. Always attach via `SessionOptions.add_provider_for_devices`.

  * `get_ep_devices()` returns BOTH a QNN NPU device and a QNN GPU device.
    Pick the NPU explicitly.

  * `get_providers()` returning 'QNNExecutionProvider' is necessary but NOT
    sufficient proof the NPU ran: individual nodes can still fall back to CPU.
    The only trustworthy signal is that the QNN HTP graph compiler emits its
    "Finalizing Graph Sequence" / "Graph Sequencing for Target" stages at
    log_severity_level <= 3. This module captures that native log during
    session build and asserts on it. A silent CPU fallback prints none of it
    (and runs ~10x slower).
"""

from __future__ import annotations

import contextlib
import os
import sys
import tempfile

import onnxruntime as ort
import onnxruntime_qnn as qnn

QNN_EP_NAME = "QNNExecutionProvider"

# Stage names the QNN HTP graph compiler prints when it actually compiles a
# subgraph for the Hexagon target. Presence of ANY of these in the native log
# is proof the op was placed on HTP, not silently dropped to CPU.
HTP_COMPILE_MARKERS = (
    "Finalizing Graph Sequence",
    "Graph Sequencing for Target",
    "VTCM Allocation",
)

_registered = False


class PlacementError(RuntimeError):
    """Raised when an NPU session did not actually compile onto the HTP."""


def register() -> None:
    """Register the onnxruntime-qnn plugin library with ONNX Runtime.

    Idempotent. Must be called before `get_ep_devices()` returns QNN devices.
    """
    global _registered
    if not _registered:
        ort.register_execution_provider_library(qnn.get_ep_name(), qnn.get_library_path())
        _registered = True


def list_qnn_devices():
    """Return every QNN OrtEpDevice (typically one NPU + one GPU)."""
    register()
    return [d for d in ort.get_ep_devices() if d.ep_name == QNN_EP_NAME]


def get_qnn_device(kind: str = "NPU"):
    """Return the QNN device of the given hardware kind ('NPU' or 'GPU')."""
    want = getattr(ort.OrtHardwareDeviceType, kind)
    devices = [d for d in list_qnn_devices() if d.device.type == want]
    if not devices:
        available = [(d.ep_name, str(d.device.type)) for d in list_qnn_devices()]
        raise RuntimeError(
            f"No QNN {kind} device found. Available QNN devices: {available}. "
            "Is onnxruntime-qnn installed and the machine a Snapdragon with HTP?"
        )
    return devices[0]


@contextlib.contextmanager
def capture_native_output():
    """Redirect OS-level stdout+stderr (fds 1 and 2) into a buffer.

    ONNX Runtime's QNN logger writes the HTP compile stages at the C level, so
    Python-level redirection (contextlib.redirect_stdout) does not catch them;
    we have to dup2 the file descriptors. Yields a dict whose "text" key holds
    the captured output once the block exits.
    """
    box = {"text": ""}
    tmp = tempfile.TemporaryFile(mode="w+b")
    saved_out, saved_err = os.dup(1), os.dup(2)
    sys.stdout.flush()
    sys.stderr.flush()
    os.dup2(tmp.fileno(), 1)
    os.dup2(tmp.fileno(), 2)
    try:
        yield box
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        os.close(saved_out)
        os.close(saved_err)
        tmp.seek(0)
        box["text"] = tmp.read().decode("utf-8", "replace")
        tmp.close()


def build_session(
    model_path: str,
    *,
    use_npu: bool = True,
    perf_mode: str = "burst",
    extra_options: dict | None = None,
    log_severity: int = 3,
    verify: bool = True,
):
    """Build an InferenceSession and (optionally) assert real HTP placement.

    Parameters
    ----------
    model_path : path to the .onnx model.
    use_npu : if True, attach the QNN NPU EP via add_provider_for_devices;
        if False, build a plain CPU-EP session (the honest baseline).
    perf_mode : QNN htp_performance_mode ("burst", "high_performance", ...).
    extra_options : extra provider options merged into the QNN attach dict.
    log_severity : ORT log severity. Must be <= 3 for the HTP compile stages
        to be emitted so verification can see them.
    verify : if True and use_npu, raise PlacementError when the HTP compile
        markers are absent (i.e. the op silently fell back to CPU).

    Returns
    -------
    (session, info) where info = {"providers", "htp_verified", "log"}.
    """
    if use_npu and verify and log_severity > 3:
        # The markers only exist in the log at severity <= 3, so a quieter
        # session cannot produce them even on a perfectly placed graph --
        # htp_verified would be False for a reason that has nothing to do with
        # placement, and this function would raise PlacementError on a correct
        # HTP run. A check whose negative result carries no information is
        # worse than no check, so refuse the combination rather than return a
        # verdict that cannot discriminate.
        raise ValueError(
            "log_severity=%d cannot be verified: the HTP compile markers are "
            "only emitted at severity <= 3, so verification would fail on a "
            "correctly placed graph. Pass log_severity <= 3, or verify=False "
            "to skip placement checking deliberately." % log_severity)
    register()
    so = ort.SessionOptions()
    so.log_severity_level = log_severity
    if use_npu:
        device = get_qnn_device("NPU")
        opts = {"htp_performance_mode": perf_mode}
        if extra_options:
            opts.update(extra_options)
        so.add_provider_for_devices([device], opts)

    with capture_native_output() as box:
        session = ort.InferenceSession(model_path, sess_options=so)
    log = box["text"]

    if use_npu:
        # By default ORT's Python wrapper silently rebuilds the session on the
        # CPU EP and retries when a QNN run raises (e.g. a transient HTP
        # "QNN_COMMON_ERROR_SYSTEM Code 1003" device error). That turns a failed
        # NPU run into a CPU result mislabelled as NPU (a fake ~1.0x speedup).
        # Disable it so run-time HTP failures surface honestly to the caller.
        session.disable_fallback()

    htp_verified = any(marker in log for marker in HTP_COMPILE_MARKERS)
    if use_npu and verify and not htp_verified:
        raise PlacementError(
            "QNN HTP compile stages were not emitted during session build -- the "
            "graph almost certainly fell back to CPU.\n"
            f"  get_providers() = {session.get_providers()}\n"
            "  Captured QNN log (first 2KB):\n"
            + "  " + (log[:2048].replace("\n", "\n  ") if log else "<empty>")
        )
    return session, {
        "providers": session.get_providers(),
        "htp_verified": htp_verified,
        "log": log,
    }


def assert_htp_placement(info: dict) -> None:
    """Raise PlacementError unless `info` (from build_session) proves HTP use."""
    if not info.get("htp_verified"):
        raise PlacementError(
            f"HTP placement not verified; providers={info.get('providers')}"
        )
