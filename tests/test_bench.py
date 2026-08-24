"""Tests for the GEMM micro-benchmark's reporting.

The measurement itself needs a Hexagon NPU and belongs in a hardware-gated
test. What is covered here is what happens when the NPU run FAILS -- the path
where `speedup` and `npu_gops` are NaN and must never reach the reader
formatted as though they were measurements. `%.1f` on a NaN prints "nan", which
in a results table looks like a number that was taken rather than one that does
not exist, and this repo has already published two figures that turned out to
be artifacts of how they were printed rather than of what ran.

`onnx` and `onnxruntime` are Snapdragon-only and absent here, so they are
stubbed at import time; nothing exercised below calls into them.
"""

import importlib
import os
import sys
import types

import pytest

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")


@pytest.fixture
def bench(monkeypatch):
    if SRC not in sys.path:
        sys.path.insert(0, SRC)
    onnx = types.ModuleType("onnx")
    onnx.TensorProto = types.SimpleNamespace(FLOAT=1, INT8=3)
    onnx.helper = types.SimpleNamespace()
    onnx.numpy_helper = types.SimpleNamespace()
    onnx.save = lambda *a, **k: None
    helper = types.ModuleType("onnx.helper")
    numpy_helper = types.ModuleType("onnx.numpy_helper")
    ort = types.ModuleType("onnxruntime")
    ort.SessionOptions = object
    ort.InferenceSession = object
    ort.OrtHardwareDeviceType = types.SimpleNamespace(NPU="NPU")
    ort.get_ep_devices = lambda: []
    ort.register_execution_provider_library = lambda *a: None
    qnn = types.ModuleType("onnxruntime_qnn")
    qnn.get_ep_name = lambda: "QNNExecutionProvider"
    qnn.get_library_path = lambda: "stub.dll"
    for name, mod in [("onnx", onnx), ("onnx.helper", helper),
                      ("onnx.numpy_helper", numpy_helper),
                      ("onnxruntime", ort), ("onnxruntime_qnn", qnn)]:
        monkeypatch.setitem(sys.modules, name, mod)
    import bench
    return importlib.reload(bench)


def _failed(**over):
    r = {"M": 512, "K": 4096, "N": 4096,
         "cpu_ms": 42.0, "cpu_gops": 204.0,
         "htp_verified": True, "providers": ["QNNExecutionProvider"],
         "unit": "GFLOP/s",
         "npu_run_error": "RuntimeError: QNN_COMMON_ERROR_SYSTEM Code 1003",
         "npu_ms": float("nan"), "npu_gops": float("nan"),
         "speedup": float("nan")}
    r.update(over)
    return r


def _ok(**over):
    r = {"M": 512, "K": 4096, "N": 4096,
         "cpu_ms": 42.0, "cpu_gops": 204.0, "npu_ms": 4.2, "npu_gops": 2040.0,
         "speedup": 10.0, "htp_verified": True,
         "providers": ["QNNExecutionProvider"], "unit": "GFLOP/s",
         "npu_run_error": None}
    r.update(over)
    return r


# --- a failed NPU run must not print as a measurement ---------------------

def test_a_failed_npu_run_prints_no_nan(bench, capsys):
    bench._print_pair(_failed())
    out = capsys.readouterr().out
    assert "nan" not in out.lower(), "a NaN in a results table reads as a number"
    assert "RUN FAILED" in out


def test_a_failed_run_names_the_device_error(bench, capsys):
    bench._print_pair(_failed())
    out = capsys.readouterr().out
    assert "1003" in out, "the caller needs the actual error, not just 'failed'"


def test_a_failed_run_still_reports_the_cpu_baseline(bench, capsys):
    # The CPU leg succeeded and is a real measurement; dropping it would waste
    # the half of the run that worked.
    bench._print_pair(_failed())
    assert "42.00" in capsys.readouterr().out


def test_a_failed_run_never_claims_a_speedup(bench, capsys):
    bench._print_pair(_failed())
    assert "speedup" not in capsys.readouterr().out.lower()


def test_a_failed_run_says_the_graph_compiled(bench, capsys):
    # htp_verified=True with an execute failure is the interesting case: the
    # placement was real and the DEVICE faltered. Conflating that with a CPU
    # fallback would send the next reader hunting the wrong bug.
    bench._print_pair(_failed())
    assert "compiled=True" in capsys.readouterr().out


# --- a successful run ------------------------------------------------------

def test_a_verified_run_reports_the_speedup(bench, capsys):
    bench._print_pair(_ok())
    out = capsys.readouterr().out
    assert "10.0x" in out and "verified" in out


def test_an_unverified_run_is_flagged_not_quietly_reported(bench, capsys):
    # The whole point of the module: a number that may have come from the CPU
    # must say so next to itself.
    bench._print_pair(_ok(htp_verified=False))
    assert "NOT VERIFIED" in capsys.readouterr().out


# --- the sweep's row formatting -------------------------------------------

def test_the_sweep_prints_FAIL_rather_than_a_nan_row(bench, capsys, monkeypatch):
    monkeypatch.setattr(bench, "make_fp32_gemm", lambda p, *a, **k: p)
    monkeypatch.setattr(bench, "_run_pair", lambda *a, **k: _failed())
    bench.bench_sweep(".", 4096, 4096, 1, 0, True, lengths=(128,))
    out = capsys.readouterr().out
    assert "FAIL" in out and "ERR" in out
    assert "nan" not in out.lower()


def test_the_sweep_marks_an_unverified_row(bench, capsys, monkeypatch):
    monkeypatch.setattr(bench, "make_fp32_gemm", lambda p, *a, **k: p)
    monkeypatch.setattr(bench, "_run_pair", lambda *a, **k: _ok(htp_verified=False))
    bench.bench_sweep(".", 4096, 4096, 1, 0, True, lengths=(128,))
    assert " NO" in capsys.readouterr().out


# --- CLI parsing ----------------------------------------------------------

def test_shape_requires_three_dimensions(bench):
    import argparse
    with pytest.raises(argparse.ArgumentTypeError):
        bench._parse_shape("512,4096")


def test_shape_parses_a_triple(bench):
    assert bench._parse_shape("512,4096,4096") == (512, 4096, 4096)
