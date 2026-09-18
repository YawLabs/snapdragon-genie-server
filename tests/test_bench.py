"""Tests for the GEMM micro-benchmark's reporting.

The measurement itself needs a Hexagon NPU and belongs in a hardware-gated
test. What is covered here is what happens when the NPU run FAILS -- the path
where `speedup` and `npu_gops` are NaN and must never reach the reader
formatted as though they were measurements. `%.1f` on a NaN prints "nan", which
in a results table looks like a number that was taken rather than one that does
not exist, and this repo has already published two figures that turned out to
be artifacts of how they were printed rather than of what ran.

`onnx` and `onnxruntime` are Snapdragon-only and absent here, so they are
stubbed at import time; nothing exercised below calls into them. The NaN
check is a WORD match (`nan`, or `nanx` as `%.1fx` prints it, not inside a
longer word): the old substring test over the whole output would have failed
on the words "tenant" or "maintenance" appearing in any message.

The last section is the other way round: a Python WITHOUT the wheels, made so
by setting them to None in sys.modules, where --help must still work and any
other run must name what is missing instead of dying on an import.
"""

import importlib
import importlib.util
import math
import os
import re
import subprocess
import sys
import types

import pytest

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")

# A formatted NaN in a table cell: "nan" or "nanx" (the speedup's `x` suffix),
# never with a letter on either side -- "tenant" and "maintenance" are words.
_NAN_CELL = re.compile(r"(?<![A-Za-z])nan(?:x)?(?![A-Za-z])")


def _nan_cells(out):
    return _NAN_CELL.findall(out)


@pytest.fixture
def bench(monkeypatch):
    if SRC not in sys.path:
        sys.path.insert(0, SRC)
    # bench.py imports bench_endpoint for its box-state sampler. Pinned to a
    # fresh load of the REAL module for the length of the test, so that what
    # bench sees never depends on which test file was collected first or on
    # what an earlier one left under that name in sys.modules. The sampler
    # itself is stubbed: it launches PowerShell.
    spec = importlib.util.spec_from_file_location(
        "bench_endpoint", os.path.join(SRC, "bench_endpoint.py"))
    be = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(be)
    be.box_state = lambda: (None, None, None, None)
    monkeypatch.setitem(sys.modules, "bench_endpoint", be)
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


_UNREAD = {"on_ac": None, "charge_pct": None, "charge_w": None, "clock_pct": None}


def _failed(**over):
    nan = float("nan")
    r = {"M": 512, "K": 4096, "N": 4096,
         "cpu_ms": 42.0, "cpu_ms_median": 41.5, "cpu_ms_min": 40.9, "cpu_gops": 204.0,
         "cpu_times_ms": [42.0, 41.5, 40.9],
         "htp_verified": True, "providers": ["QNNExecutionProvider"],
         "unit": "GFLOP/s",
         "npu_run_error": "RuntimeError: QNN_COMMON_ERROR_SYSTEM Code 1003",
         "npu_ms": nan, "npu_ms_median": nan, "npu_ms_min": nan, "npu_gops": nan,
         "npu_times_ms": [], "speedup": nan, "speedup_median": nan,
         "box": dict(_UNREAD)}
    r.update(over)
    return r


def _ok(**over):
    r = {"M": 512, "K": 4096, "N": 4096,
         "cpu_ms": 42.0, "cpu_ms_median": 41.5, "cpu_ms_min": 40.9, "cpu_gops": 204.0,
         "cpu_times_ms": [42.0, 41.5, 40.9],
         "npu_ms": 4.2, "npu_ms_median": 4.1, "npu_ms_min": 4.0, "npu_gops": 2040.0,
         "npu_times_ms": [4.2, 4.1, 4.0],
         "speedup": 10.0, "speedup_median": 10.1, "htp_verified": True,
         "providers": ["QNNExecutionProvider"], "unit": "GFLOP/s",
         "npu_run_error": None, "box": dict(_UNREAD)}
    r.update(over)
    return r


# --- the producer of that shape ---------------------------------------------
# _failed()/_ok() hand-build the dict, so a key rename in _run_pair would leave
# every printing test below reading the OLD key from a literal and passing.
# This runs the real producer over a stubbed session so the shape is pinned
# where it is made.

class _Clock:
    """A perf_counter that only a session's run() advances.

    The stubbed sessions return at once, and two real perf_counter reads around
    a no-op can land inside one 100 ns tick: a zero-second timing, then a
    ZeroDivisionError out of the GFLOP/s arithmetic that has nothing to do with
    the code under test. A scripted clock also lets the three summaries be
    asserted as numbers rather than as inequalities.
    """

    def __init__(self):
        self.now = 100.0

    def perf_counter(self):
        return self.now


class _Session:
    """run() takes `seconds[i]` of the scripted clock on its i-th call, cycling."""

    def __init__(self, clock, seconds=(0.004,), raises=None):
        self.clock, self.seconds, self.raises, self.runs = clock, seconds, raises, 0

    def run(self, _outputs, _feeds):
        self.runs += 1
        if self.raises is not None:
            raise self.raises
        self.clock.now += self.seconds[(self.runs - 1) % len(self.seconds)]


def _stub_sessions(bench, monkeypatch, npu_seconds=(0.004,), npu_raises=None):
    """(npu, cpu) sessions behind qnn_ep.build_session: 4 ms and 40 ms a run."""
    clock = _Clock()
    monkeypatch.setattr(bench, "time", clock)
    npu = _Session(clock, npu_seconds, npu_raises)
    cpu = _Session(clock, (0.040,))

    def build_session(path, use_npu, verify):
        sess = npu if use_npu else cpu
        return sess, {"htp_verified": use_npu, "providers": ["QNNExecutionProvider"]}
    monkeypatch.setattr(bench.qnn_ep, "build_session", build_session)
    return npu, cpu


def test_run_pair_reports_a_failed_npu_run_as_nans_under_the_printed_keys(bench, monkeypatch):
    # Exception("") on purpose: an empty message once raised IndexError from
    # INSIDE the handler and replaced the device error with one from the
    # error path.
    _, cpu = _stub_sessions(bench, monkeypatch, npu_raises=Exception(""))
    r = bench._run_pair("m.onnx", {}, 512, 4096, 4096, iters=3, warmup=1, verify=True,
                        unit="GFLOP/s")
    assert r["npu_run_error"] == "Exception: "
    for k in ("npu_ms", "npu_ms_median", "npu_ms_min", "npu_gops", "speedup",
              "speedup_median"):
        assert math.isnan(r[k]), k
    assert r["npu_times_ms"] == []
    assert cpu.runs == 4, "the CPU leg still ran (1 warmup + 3 timed)"
    assert r["cpu_ms"] == pytest.approx(40.0), "and is still a real measurement"
    assert set(_failed()) == set(r), "the hand-built fixture must match the producer"


def test_run_pair_carries_per_iteration_times_and_the_three_summaries(bench, monkeypatch):
    # 4, 5 and 9 ms: one slow iteration, the co-tenant spike. It moves the
    # mean to 6 and leaves the median at 5 and the min at 4.
    _stub_sessions(bench, monkeypatch, npu_seconds=(0.004, 0.005, 0.009))
    r = bench._run_pair("m.onnx", {}, 512, 4096, 4096, iters=3, warmup=0, verify=True,
                        unit="GFLOP/s")
    assert r["npu_run_error"] is None
    assert r["npu_times_ms"] == pytest.approx([4.0, 5.0, 9.0])
    assert r["cpu_times_ms"] == pytest.approx([40.0, 40.0, 40.0])
    assert (r["npu_ms"], r["npu_ms_median"], r["npu_ms_min"]) == pytest.approx((6.0, 5.0, 4.0))
    assert r["speedup"] == pytest.approx(40.0 / 6.0), "the headline stays the MEAN's"
    assert r["speedup_median"] == pytest.approx(8.0)
    assert r["npu_gops"] == pytest.approx(2.0 * 512 * 4096 * 4096 / 0.006 / 1e9)
    assert set(_ok()) == set(r), "the hand-built fixture must match the producer"


def test_run_pair_records_the_box_state_under_bench_endpoints_keys(bench, monkeypatch):
    _stub_sessions(bench, monkeypatch)
    monkeypatch.setattr(bench.bench_endpoint, "box_state", lambda: (True, 73.0, 12.5, 88.0))
    r = bench._run_pair("m.onnx", {}, 8, 8, 8, iters=1, warmup=0, verify=True, unit="GFLOP/s")
    assert r["box"] == {"on_ac": True, "charge_pct": 73.0, "charge_w": 12.5, "clock_pct": 88.0}


def test_run_pair_survives_a_sampler_that_raises(bench, monkeypatch):
    _stub_sessions(bench, monkeypatch)

    def boom():
        raise RuntimeError("powershell went away")
    monkeypatch.setattr(bench.bench_endpoint, "box_state", boom)
    r = bench._run_pair("m.onnx", {}, 8, 8, 8, iters=1, warmup=0, verify=True, unit="GFLOP/s")
    assert r["box"] == _UNREAD


def test_only_the_npu_leg_is_placement_verified(bench, monkeypatch):
    # The last hop of --no-verify. The CPU leg is a CPU number on purpose, so
    # it is built with the check off either way; the NPU leg is the one where
    # an unverified placement means a CPU number labelled "NPU".
    seen = []
    clock = _Clock()
    monkeypatch.setattr(bench, "time", clock)
    npu, cpu = _Session(clock), _Session(clock, (0.040,))

    def build_session(path, use_npu, verify):
        seen.append((use_npu, verify))
        return (npu if use_npu else cpu), {"htp_verified": use_npu, "providers": []}
    monkeypatch.setattr(bench.qnn_ep, "build_session", build_session)
    bench._run_pair("m.onnx", {}, 8, 8, 8, iters=1, warmup=0, verify=True, unit="GFLOP/s")
    assert seen == [(True, True), (False, False)]
    seen.clear()
    bench._run_pair("m.onnx", {}, 8, 8, 8, iters=1, warmup=0, verify=False, unit="GFLOP/s")
    assert seen == [(True, False), (False, False)], "--no-verify reaches the build"


# --- the two model builders -------------------------------------------------
# These define what every published GFLOP/s and GOP/s figure is a figure OF,
# and nothing has ever called them: the fixture's onnx stub has an empty
# helper, so either builder would AttributeError. A wrong quant scale or a
# broken QDQ pattern still compiles and still times, so the number arrives
# looking normal. Recorded rather than executed -- whether the HTP genuinely
# FUSES the pattern is hardware-gated and belongs in a device test; the graph
# it is handed is not.

class _Onnx:
    """A recording stand-in for the onnx module the builders call."""

    def __init__(self):
        self.saved = []

    def _ns(self, **kw):
        return types.SimpleNamespace(**kw)

    @property
    def helper(self):
        return types.SimpleNamespace(
            make_node=lambda op, ins, outs: self._ns(
                op_type=op, input=list(ins), output=list(outs)),
            make_tensor=lambda name, dtype, dims, vals: self._ns(
                name=name, data_type=dtype, dims=list(dims), vals=vals),
            make_tensor_value_info=lambda name, dtype, shape: self._ns(
                name=name, data_type=dtype, shape=list(shape)),
            make_opsetid=lambda domain, version: self._ns(
                domain=domain, version=version),
            make_graph=lambda nodes, name, ins, outs, inits: self._ns(
                node=list(nodes), name=name, input=list(ins), output=list(outs),
                initializer=list(inits)),
            make_model=lambda graph, opset_imports: self._ns(
                graph=graph, opset_import=list(opset_imports), ir_version=None),
        )

    @property
    def numpy_helper(self):
        return types.SimpleNamespace(
            from_array=lambda arr, name: self._ns(name=name, array=arr))

    def save(self, model, path):
        self.saved.append((path, model))


@pytest.fixture
def onnx_calls(bench, monkeypatch):
    rec = _Onnx()
    monkeypatch.setattr(bench, "onnx", rec)
    monkeypatch.setattr(bench, "helper", rec.helper)
    monkeypatch.setattr(bench, "numpy_helper", rec.numpy_helper)
    return rec


def _init(model, name):
    (found,) = [i for i in model.graph.initializer if i.name == name]
    return found


def test_the_fp32_gemm_is_one_matmul_over_the_asked_for_shape(bench, onnx_calls):
    assert bench.make_fp32_gemm("m.onnx", 512, 4096, 2048) == "m.onnx"
    ((path, m),) = onnx_calls.saved
    assert path == "m.onnx"
    assert [(n.op_type, n.input, n.output) for n in m.graph.node] == [
        ("MatMul", ["A", "W"], ["Y"])]
    (a,), (y,) = m.graph.input, m.graph.output
    assert (a.name, a.shape, a.data_type) == ("A", [512, 4096], bench.TensorProto.FLOAT)
    assert (y.name, y.shape, y.data_type) == ("Y", [512, 2048], bench.TensorProto.FLOAT)
    w = _init(m, "W")
    assert w.dims == [4096, 2048] and len(w.vals) == 4096 * 2048
    assert w.data_type == bench.TensorProto.FLOAT, "FP32 IO; the HTP runs it in FP16"


def test_the_fp32_gemm_is_built_at_the_opset_and_ir_version_the_htp_takes(
        bench, onnx_calls):
    bench.make_fp32_gemm("m.onnx", 8, 8, 8)
    ((_path, m),) = onnx_calls.saved
    assert [(o.domain, o.version) for o in m.opset_import] == [("", 17)]
    assert m.ir_version == 10, "the default is past what this ORT build reads"


def test_the_qdq_gemm_is_the_pattern_the_htp_fuses(bench, onnx_calls):
    # Quantize the activation, dequantize both sides, THEN matmul. A MatMul
    # over the quantized tensors instead is a different computation that still
    # builds, still times and still prints a GOP/s figure.
    bench.make_qdq_gemm("q.onnx", 512, 256, 128)
    ((_path, m),) = onnx_calls.saved
    assert [(n.op_type, n.input, n.output) for n in m.graph.node] == [
        ("QuantizeLinear", ["A", "A_s", "A_z"], ["Aq"]),
        ("DequantizeLinear", ["Aq", "A_s", "A_z"], ["Ad"]),
        ("DequantizeLinear", ["Wq", "W_s", "W_z"], ["Wd"]),
        ("MatMul", ["Ad", "Wd"], ["Y"])]
    assert [(o.domain, o.version) for o in m.opset_import] == [("", 21)]
    assert m.ir_version == 10


def test_the_qdq_weights_are_symmetric_int8_at_a_per_tensor_scale(bench, onnx_calls):
    # The scale is max|W|/127, so the largest weight lands exactly at 127 and
    # nothing saturates; the zero points are 0, which is what makes it
    # symmetric. A scale off by 2x still quantizes, still runs and reports a
    # number in the same column as a good one.
    K, N = 64, 32
    # The builder's own weights, regenerated from its seed -- the only way to
    # say what the scale is a scale OF.
    Wf = bench.np.random.default_rng(0).standard_normal((K, N)).astype("float32") * 0.02
    bench.make_qdq_gemm("q.onnx", 8, K, N)
    ((_path, m),) = onnx_calls.saved
    scale = _init(m, "W_s").vals[0]
    Wq = _init(m, "Wq").array
    assert str(Wq.dtype) == "int8" and Wq.shape == (K, N)
    assert float(abs(Wf).max()) / scale == pytest.approx(127.0, rel=1e-6)
    assert int(abs(Wq).max()) == 127, "the top of the range is used, and not passed"
    assert float(abs(Wq * scale - Wf).max()) <= scale / 2 * (1 + 1e-6), "half a step"
    for name in ("W_z", "A_z"):
        z = _init(m, name)
        assert z.vals == [0] and z.data_type == bench.TensorProto.INT8


def test_the_qdq_activation_scale_is_the_fixed_symmetric_one(bench, onnx_calls):
    # Fixed, not derived from the random A: the activation is regenerated per
    # run, and a scale that moved with it would make two runs of the same
    # shape two different computations.
    bench.make_qdq_gemm("q.onnx", 8, 8, 8)
    ((_path, m),) = onnx_calls.saved
    s = _init(m, "A_s")
    assert s.vals == [0.05] and s.data_type == bench.TensorProto.FLOAT


# --- per-iteration timing ---------------------------------------------------

def test_time_session_keeps_one_sample_per_iteration_after_warmup(bench, monkeypatch):
    npu, _ = _stub_sessions(bench, monkeypatch, npu_seconds=(0.001, 0.002))
    times = bench.time_session(npu, {}, iters=5, warmup=2)
    assert npu.runs == 7
    # One sample per timed run, the two warmup runs in none of them.
    assert times == pytest.approx([0.001, 0.002, 0.001, 0.002, 0.001])


def test_summarise_is_mean_median_min(bench):
    assert bench.summarise([3.0, 1.0, 2.0, 10.0]) == (4.0, 2.5, 1.0)


# --- a failed NPU run must not print as a measurement ---------------------

def test_a_failed_npu_run_prints_no_nan(bench, capsys):
    bench._print_pair(_failed())
    out = capsys.readouterr().out
    assert _nan_cells(out) == [], "a NaN in a results table reads as a number"
    assert "RUN FAILED" in out


def test_the_nan_check_matches_a_formatted_nan_and_not_a_word():
    assert _nan_cells("  NPU/HTP       nan ms/run  nanx") == ["nan", "nanx"]
    assert _nan_cells("co-tenant maintenance unmanaged") == []


def test_a_failed_run_prints_no_median_or_min_for_the_npu_leg(bench, capsys):
    bench._print_pair(_failed())
    npu_lines = [ln for ln in capsys.readouterr().out.splitlines() if "NPU/HTP" in ln]
    assert npu_lines and all("median" not in ln for ln in npu_lines)


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


def test_a_verified_run_prints_median_and_min_beside_the_mean(bench, capsys):
    bench._print_pair(_ok())
    out = capsys.readouterr().out
    assert "median 4.10" in out and "min 4.00" in out, "the NPU leg's spread"
    assert "median 41.50" in out and "min 40.90" in out, "the CPU leg's spread"
    assert "10.1x by median" in out


# --- box state beside every case --------------------------------------------

def test_an_unread_box_state_says_unreadable_not_zero(bench, capsys):
    bench._print_pair(_ok())
    out = capsys.readouterr().out
    assert "unreadable" in out
    assert "clock 0%" not in out and "-1%" not in out


def test_a_read_box_state_prints_clock_pack_and_draw(bench, capsys):
    bench._print_pair(_ok(box={"on_ac": True, "charge_pct": 73.0, "charge_w": 12.5,
                               "clock_pct": 88.0}))
    out = capsys.readouterr().out
    assert "clock 88% of base" in out and "pack 73% on AC" in out and "draw 12.5 W" in out


def test_a_battery_run_is_named_as_such(bench, capsys):
    bench._print_pair(_ok(box={"on_ac": False, "charge_pct": 40.0, "charge_w": None,
                               "clock_pct": None}))
    assert "ON BATTERY" in capsys.readouterr().out


def test_the_clock_cell_is_dashes_when_unread(bench):
    assert bench._clock_cell(_UNREAD) == "--"
    assert bench._clock_cell({"clock_pct": 87.6}) == "88"
    assert bench._clock_cell(None) == "--"


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
    row = [ln for ln in out.splitlines() if ln.strip().startswith("128")]
    assert row, "no row for M=128"
    # The exact cells, not a substring hunt: NPU ms, NPU median, GFLOP/s and
    # the win are all "--"/FAIL, the CPU figure is real, the flag is ERR.
    assert row[0].split() == ["128", "FAIL", "--", "42.00", "--", "--", "ERR", "--"]
    assert _nan_cells(out) == []


def test_the_sweep_marks_an_unverified_row(bench, capsys, monkeypatch):
    monkeypatch.setattr(bench, "make_fp32_gemm", lambda p, *a, **k: p)
    monkeypatch.setattr(bench, "_run_pair", lambda *a, **k: _ok(htp_verified=False))
    bench.bench_sweep(".", 4096, 4096, 1, 0, True, lengths=(128,))
    assert " NO" in capsys.readouterr().out


def test_the_sweep_row_carries_the_median_and_the_clock(bench, capsys, monkeypatch):
    monkeypatch.setattr(bench, "make_fp32_gemm", lambda p, *a, **k: p)
    monkeypatch.setattr(bench, "_run_pair",
                        lambda *a, **k: _ok(box={**_UNREAD, "clock_pct": 91.0}))
    bench.bench_sweep(".", 4096, 4096, 1, 0, True, lengths=(128,))
    out = capsys.readouterr().out
    row = [ln for ln in out.splitlines() if ln.strip().startswith("128")]
    assert row[0].split() == ["128", "4.20", "4.10", "42.00", "2040.0", "10.0x", "yes", "91"]
    assert "NPU med" in out and "clk%" in out


# --- CLI parsing ----------------------------------------------------------

def test_shape_requires_three_dimensions(bench):
    import argparse
    with pytest.raises(argparse.ArgumentTypeError):
        bench._parse_shape("512,4096")


def test_shape_parses_a_triple(bench):
    assert bench._parse_shape("512,4096,4096") == (512, 4096, 4096)


# --- the CLI's wiring -------------------------------------------------------
# main() takes `argv` for exactly this and nothing had ever passed one. The
# three things it decides are all silent when wrong: which cases run, whether a
# silent CPU fallback is a hard failure or a slow success -- the module
# docstring's core promise -- and which two of M,K,N the sweep is a sweep over.

def _cases(bench, monkeypatch):
    """What main() ran, and what it handed each case."""
    ran = {}

    def pair(name):
        def case(model_dir, shape, iters, warmup, verify):
            ran[name] = {"model_dir": model_dir, "shape": shape, "iters": iters,
                         "warmup": warmup, "verify": verify}
        return case

    def sweep(model_dir, K, N, iters, warmup, verify):
        ran["sweep"] = {"model_dir": model_dir, "K": K, "N": N, "iters": iters,
                        "warmup": warmup, "verify": verify}

    monkeypatch.setattr(bench, "bench_fp16", pair("fp16"))
    monkeypatch.setattr(bench, "bench_int8", pair("int8"))
    monkeypatch.setattr(bench, "bench_sweep", sweep)
    monkeypatch.setattr(bench.qnn_ep, "list_qnn_devices", lambda: [])
    monkeypatch.setattr(sys.modules["onnxruntime"], "__version__", "1.23.0",
                        raising=False)
    return ran


def test_a_bare_run_does_every_case(bench, monkeypatch, tmp_path, capsys):
    # `python bench.py` with no case flag. Losing the default turns it into a
    # header and nothing else -- a run that looks like it did the work.
    ran = _cases(bench, monkeypatch)
    bench.main(["--model-dir", str(tmp_path)])
    assert sorted(ran) == ["fp16", "int8", "sweep"]
    assert "onnxruntime 1.23.0" in capsys.readouterr().out


def test_a_named_case_is_the_only_one_that_runs(bench, monkeypatch, tmp_path):
    ran = _cases(bench, monkeypatch)
    bench.main(["--int8", "--model-dir", str(tmp_path)])
    assert sorted(ran) == ["int8"]


def test_placement_verification_is_on_until_no_verify_turns_it_off(
        bench, monkeypatch, tmp_path, capsys):
    # verify is what makes a silent CPU fallback a hard failure rather than a
    # slow success. Inverted, the tool publishes CPU numbers labelled NPU.
    ran = _cases(bench, monkeypatch)
    bench.main(["--fp16", "--model-dir", str(tmp_path)])
    assert ran["fp16"]["verify"] is True
    assert "verify=True" in capsys.readouterr().out, "it is in the header too"
    ran.clear()
    bench.main(["--fp16", "--no-verify", "--model-dir", str(tmp_path)])
    assert ran["fp16"]["verify"] is False
    assert "verify=False" in capsys.readouterr().out


def test_the_sweep_is_handed_the_K_and_the_N_not_the_M(bench, monkeypatch, tmp_path):
    # The sweep varies M itself, so it takes the other two. Handed M,K it
    # would sweep a differently-shaped GEMM than --shape asked for and report
    # it under the same name.
    ran = _cases(bench, monkeypatch)
    bench.main(["--sweep", "--shape", "512,4096,2048", "--model-dir", str(tmp_path)])
    assert (ran["sweep"]["K"], ran["sweep"]["N"]) == (4096, 2048)


def test_the_cases_get_the_iteration_counts_and_the_model_dir_asked_for(
        bench, monkeypatch, tmp_path):
    ran = _cases(bench, monkeypatch)
    bench.main(["--all", "--shape", "8,16,32", "--iters", "5", "--warmup", "1",
                "--model-dir", str(tmp_path)])
    assert ran["fp16"] == {"model_dir": str(tmp_path), "shape": (8, 16, 32),
                           "iters": 5, "warmup": 1, "verify": True}
    assert ran["int8"]["shape"] == (8, 16, 32)
    assert (ran["sweep"]["iters"], ran["sweep"]["warmup"]) == (5, 1)


# --- a Python without the wheels --------------------------------------------
# bench.py imported numpy and onnx unguarded, ahead of argparse: a shell
# without the venv active got `ModuleNotFoundError: No module named 'onnx'`
# for any command, --help included, while qnn_ep's guard three lines down
# already said what to do for a missing onnxruntime. A module set to None in
# sys.modules is one that will not import, which is how each test below makes
# its Python lack a package -- whatever this Python actually has installed.

ALL_WHEELS = ("numpy", "onnx", "onnx.helper", "onnx.numpy_helper",
              "onnxruntime", "onnxruntime_qnn")


@pytest.fixture
def bench_without(bench, monkeypatch):
    """bench.py as a Python lacking `names` imports it."""
    def load(*names):
        for name in names:
            monkeypatch.setitem(sys.modules, name, None)
        # Re-imported, so its own guard runs against what is missing now.
        monkeypatch.delitem(sys.modules, "qnn_ep", raising=False)
        return importlib.reload(bench)
    return load


def test_help_works_without_any_of_the_wheels(bench_without, capsys):
    b = bench_without(*ALL_WHEELS)
    with pytest.raises(SystemExit) as e:
        b.main(["--help"])
    assert e.value.code == 0
    out = capsys.readouterr().out
    assert "usage:" in out and "--sweep" in out


def test_a_usage_error_is_still_argparse_s_without_the_wheels(bench_without, capsys):
    b = bench_without(*ALL_WHEELS)
    with pytest.raises(SystemExit) as e:
        b.main(["--shape", "1,2"])
    assert e.value.code == 2
    assert "shape must be M,K,N" in capsys.readouterr().err


def test_a_run_without_onnx_names_it_and_the_install_and_does_nothing(
        bench_without, tmp_path):
    b = bench_without("onnx", "onnx.helper", "onnx.numpy_helper")
    model_dir = tmp_path / "models"
    with pytest.raises(SystemExit) as e:
        b.main(["--sweep", "--model-dir", str(model_dir)])
    said = str(e.value)
    assert "onnx --" in said
    assert "requirements.txt" in said and "README, Install" in said
    assert sys.executable in said, "the interpreter it ran under: usually the wrong one"
    assert "numpy --" not in said, "only what is actually missing"
    assert not model_dir.exists(), "refused before anything was made"


def test_every_missing_package_is_named_at_once(bench_without):
    b = bench_without(*ALL_WHEELS)
    with pytest.raises(SystemExit) as e:
        b.main([])
    said = str(e.value)
    for package in ("numpy --", "onnx --", "onnxruntime-qnn --"):
        assert package in said, package
    # qnn_ep's own sentence is carried whole, not replaced.
    assert "onnxruntime-qnn wheel" in said


def test_the_report_is_one_line_per_package_under_one_heading(bench):
    said = bench.missing_report([("onnx", ImportError("No module named 'onnx'"))],
                                python="/opt/py/bin/python3")
    lines = said.splitlines()
    assert lines[0].startswith("bench.py needs packages") and "/opt/py/bin/python3" in lines[0]
    assert lines[1] == "  onnx -- No module named 'onnx'"
    assert "requirements.txt" in lines[-1]


def test_the_command_line_says_so_too_with_no_traceback(tmp_path):
    # The real entry point, `if __name__ == "__main__"` and all, in a child
    # Python whose wheels are made unimportable the same way. Neither run gets
    # past its imports' report, so nothing here touches a device.
    script = os.path.join(SRC, "bench.py")
    boot = ("import runpy, sys; sys.modules.update(dict.fromkeys(%r)); "
            "sys.argv = [%r] + sys.argv[1:]; runpy.run_path(%r, run_name='__main__')"
            % (list(ALL_WHEELS), script, script))
    help_run = subprocess.run([sys.executable, "-c", boot, "--help"],
                              capture_output=True, text=True, timeout=60)
    assert help_run.returncode == 0, help_run.stderr
    assert "usage:" in help_run.stdout
    run = subprocess.run([sys.executable, "-c", boot, "--sweep", "--model-dir",
                          str(tmp_path / "m")], capture_output=True, text=True, timeout=60)
    assert run.returncode == 1
    assert "requirements.txt" in run.stderr
    assert "Traceback" not in run.stderr
    assert not (tmp_path / "m").exists()
