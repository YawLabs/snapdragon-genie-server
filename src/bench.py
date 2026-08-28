"""Single-GEMM micro-benchmark for the Snapdragon X Elite Hexagon NPU (HTP)
via ONNX Runtime's QNN Execution Provider, compared against the ORT CPU EP.

What this measures
------------------
The dominant primitive in LLM *prefill* is a big batched GEMM:
[M tokens, K] x [K, N]. This tool builds such GEMMs as ONNX models and times
them on the HTP vs the CPU EP. It is NOT a full-model tokens/sec benchmark --
it isolates the matmul primitive so the NPU/CPU delta is clean and honest.

Three cases:
  * fp16 : FP32-IO MatMul (HTP runs it internally in FP16).
  * int8 : QDQ INT8 MatMul (QuantizeLinear/DequantizeLinear + MatMul; HTP
           fuses this into an int8 matmul).
  * sweep: FP32 MatMul across prompt lengths M = 128 / 512 / 2048.

Every NPU run is placement-verified: session build is only accepted if the
QNN HTP graph compiler actually emitted its compile stages (see qnn_ep.py).
A silent CPU fallback is treated as a failure, not a slow success.

Usage
-----
    python bench.py --all
    python bench.py --fp16 --shape 512,4096,4096 --iters 30
    python bench.py --int8
    python bench.py --sweep
    python bench.py --all --no-verify      # do not hard-fail on CPU fallback
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qnn_ep


# --------------------------------------------------------------------------
# Model builders
# --------------------------------------------------------------------------

def make_fp32_gemm(path: str, M: int, K: int, N: int, seed: int = 0) -> str:
    """A plain FP32 MatMul [M,K] x [K,N]. HTP executes it in FP16."""
    rng = np.random.default_rng(seed)
    W = (rng.standard_normal((K, N)).astype(np.float32) * 0.02)
    g = helper.make_graph(
        [helper.make_node("MatMul", ["A", "W"], ["Y"])],
        "fp32_gemm",
        [helper.make_tensor_value_info("A", TensorProto.FLOAT, [M, K])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [M, N])],
        [helper.make_tensor("W", TensorProto.FLOAT, [K, N], W.flatten())],
    )
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 10
    onnx.save(m, path)
    return path


def make_qdq_gemm(path: str, M: int, K: int, N: int, seed: int = 0) -> str:
    """A QDQ INT8 MatMul: quantize A, dequant A and W, MatMul. HTP fuses to int8."""
    rng = np.random.default_rng(seed)
    Wf = (rng.standard_normal((K, N)).astype(np.float32) * 0.02)
    sW = float(np.abs(Wf).max() / 127.0)
    Wq = np.clip(np.round(Wf / sW), -127, 127).astype(np.int8)
    sA = 0.05  # fixed symmetric activation scale

    def qdq(name, scale):
        return [
            helper.make_tensor(f"{name}_s", TensorProto.FLOAT, [], [scale]),
            helper.make_tensor(f"{name}_z", TensorProto.INT8, [], [0]),
        ]

    nodes = [
        helper.make_node("QuantizeLinear", ["A", "A_s", "A_z"], ["Aq"]),
        helper.make_node("DequantizeLinear", ["Aq", "A_s", "A_z"], ["Ad"]),
        helper.make_node("DequantizeLinear", ["Wq", "W_s", "W_z"], ["Wd"]),
        helper.make_node("MatMul", ["Ad", "Wd"], ["Y"]),
    ]
    inits = [numpy_helper.from_array(Wq, "Wq"), *qdq("A", sA), *qdq("W", sW)]
    g = helper.make_graph(
        nodes, "qdq_gemm",
        [helper.make_tensor_value_info("A", TensorProto.FLOAT, [M, K])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [M, N])],
        inits,
    )
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 21)])
    m.ir_version = 10
    onnx.save(m, path)
    return path


# --------------------------------------------------------------------------
# Timing
# --------------------------------------------------------------------------

def time_session(session, feeds: dict, iters: int, warmup: int) -> float:
    """Return mean seconds/run over `iters` runs after `warmup` warmup runs."""
    for _ in range(warmup):
        session.run(None, feeds)
    t0 = time.perf_counter()
    for _ in range(iters):
        session.run(None, feeds)
    return (time.perf_counter() - t0) / iters


def _run_pair(model_path, feeds, M, K, N, iters, warmup, verify, unit):
    """Build+time on NPU (verified) and CPU; return a result dict.

    HTP placement is checked twice: at compile time (marker in the QNN log,
    via build_session) and at run time (the NPU run either completes on HTP or
    raises -- fallback is disabled, so a transient HTP execute error like
    QNN_COMMON_ERROR_SYSTEM Code 1003 surfaces here instead of silently
    producing a CPU number labelled "NPU").
    """
    ops = 2.0 * M * K * N  # MAC counted as 2 ops
    npu_sess, npu_info = qnn_ep.build_session(model_path, use_npu=True, verify=verify)
    npu_s = None
    npu_run_error = None
    try:
        npu_s = time_session(npu_sess, feeds, iters, warmup)
    except Exception as e:  # HTP execute failed; fallback is disabled
        # `or [""]` because splitlines() on an empty message returns [] -- an
        # exception with no text would then raise IndexError from inside this
        # handler and replace the device error with one from the error path.
        npu_run_error = f"{type(e).__name__}: {(str(e).splitlines() or [''])[0][:180]}"

    cpu_sess, _ = qnn_ep.build_session(model_path, use_npu=False, verify=False)
    cpu_s = time_session(cpu_sess, feeds, iters, warmup)

    r = {
        "M": M, "K": K, "N": N,
        "cpu_ms": cpu_s * 1e3, "cpu_gops": ops / cpu_s / 1e9,
        "htp_verified": npu_info["htp_verified"],
        "providers": npu_info["providers"],
        "unit": unit,
        "npu_run_error": npu_run_error,
    }
    if npu_run_error is None:
        r.update(npu_ms=npu_s * 1e3, npu_gops=ops / npu_s / 1e9, speedup=cpu_s / npu_s)
    else:
        r.update(npu_ms=float("nan"), npu_gops=float("nan"), speedup=float("nan"))
    return r


# --------------------------------------------------------------------------
# Benchmark cases
# --------------------------------------------------------------------------

def bench_fp16(model_dir, shape, iters, warmup, verify):
    M, K, N = shape
    path = make_fp32_gemm(os.path.join(model_dir, f"fp32_{M}_{K}_{N}.onnx"), M, K, N)
    A = (np.random.default_rng(1).standard_normal((M, K)).astype(np.float32) * 0.05)
    r = _run_pair(path, {"A": A}, M, K, N, iters, warmup, verify, "GFLOP/s")
    print(f"\n=== FP16 GEMM {M}x{K}x{N} (FP32 IO, HTP runs FP16) ===")
    _print_pair(r)
    return r


def bench_int8(model_dir, shape, iters, warmup, verify):
    M, K, N = shape
    path = make_qdq_gemm(os.path.join(model_dir, f"qdq_{M}_{K}_{N}.onnx"), M, K, N)
    A = (np.random.default_rng(1).standard_normal((M, K)).astype(np.float32) * 0.05)
    r = _run_pair(path, {"A": A}, M, K, N, iters, warmup, verify, "GOP/s")
    print(f"\n=== INT8 QDQ GEMM {M}x{K}x{N} (HTP fuses to int8) ===")
    _print_pair(r)
    return r


def bench_sweep(model_dir, K, N, iters, warmup, verify, lengths=(128, 512, 2048)):
    print(f"\n=== FP16 prompt-length sweep (K=N={K}) ===")
    print(f"{'tokens':>7} {'NPU ms':>8} {'CPU ms':>8} {'NPU GFLOP/s':>12} {'NPU win':>8} {'HTP':>4}")
    results = []
    for M in lengths:
        path = make_fp32_gemm(os.path.join(model_dir, f"sweep_{M}.onnx"), M, K, N)
        A = (np.random.default_rng(1).standard_normal((M, K)).astype(np.float32) * 0.05)
        r = _run_pair(path, {"A": A}, M, K, N, iters, warmup, verify, "GFLOP/s")
        if r["npu_run_error"]:
            print(f"{M:>7} {'FAIL':>8} {r['cpu_ms']:>8.2f} {'--':>12} {'--':>8} {'ERR':>4}")
        else:
            ok = "yes" if r["htp_verified"] else "NO"
            print(f"{M:>7} {r['npu_ms']:>8.2f} {r['cpu_ms']:>8.2f} "
                  f"{r['npu_gops']:>12.1f} {r['speedup']:>7.1f}x {ok:>4}")
        results.append(r)
    return results


def _print_pair(r):
    if r["npu_run_error"]:
        print(f"  NPU/HTP  RUN FAILED: {r['npu_run_error']}")
        print(f"           (HTP graph compiled={r['htp_verified']}, but execute failed -- "
              "transient device error, NOT reported as a CPU number)")
    else:
        ok = "verified" if r["htp_verified"] else "NOT VERIFIED (CPU fallback?)"
        print(f"  NPU/HTP  {r['npu_ms']:8.2f} ms/run  {r['npu_gops']:8.1f} {r['unit']}   [HTP {ok}]")
    print(f"  CPU EP   {r['cpu_ms']:8.2f} ms/run  {r['cpu_gops']:8.1f} {r['unit']}")
    if not r["npu_run_error"]:
        print(f"  NPU speedup: {r['speedup']:.1f}x   providers={r['providers']}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _parse_shape(s: str):
    parts = [int(x) for x in s.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("shape must be M,K,N")
    return tuple(parts)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fp16", action="store_true", help="FP16 GEMM case")
    ap.add_argument("--int8", action="store_true", help="INT8 QDQ GEMM case")
    ap.add_argument("--sweep", action="store_true", help="FP16 prompt-length sweep")
    ap.add_argument("--all", action="store_true", help="run all cases (default)")
    ap.add_argument("--shape", type=_parse_shape, default=(512, 4096, 4096),
                    help="M,K,N for fp16/int8 (default 512,4096,4096)")
    ap.add_argument("--iters", type=int, default=30, help="timed iterations")
    ap.add_argument("--warmup", type=int, default=3, help="warmup iterations")
    ap.add_argument("--model-dir", default=None,
                    help="where to write generated .onnx (default: a temp dir)")
    ap.add_argument("--no-verify", action="store_true",
                    help="do not hard-fail if HTP placement can't be verified")
    args = ap.parse_args(argv)

    if not (args.fp16 or args.int8 or args.sweep):
        args.all = True
    verify = not args.no_verify

    model_dir = args.model_dir or os.path.join(tempfile.gettempdir(), "snpu_bench_models")
    os.makedirs(model_dir, exist_ok=True)

    devs = [(d.ep_name, str(d.device.type)) for d in qnn_ep.list_qnn_devices()]
    print(f"onnxruntime {__import__('onnxruntime').__version__}  "
          f"onnxruntime_qnn {getattr(__import__('onnxruntime_qnn'), '__version__', 'n/a')}  "
          f"python {sys.version.split()[0]}")
    print(f"QNN devices: {devs}")
    print(f"model dir: {model_dir}   iters={args.iters} warmup={args.warmup} verify={verify}")

    if args.all or args.fp16:
        bench_fp16(model_dir, args.shape, args.iters, args.warmup, verify)
    if args.all or args.int8:
        bench_int8(model_dir, args.shape, args.iters, args.warmup, verify)
    if args.all or args.sweep:
        _, K, N = args.shape
        bench_sweep(model_dir, K, N, args.iters, args.warmup, verify)


if __name__ == "__main__":
    main()
