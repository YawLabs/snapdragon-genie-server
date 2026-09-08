#!/usr/bin/env python3
"""Interleaved decode-rate A/B between two OpenAI-compatible servers.

`bench_endpoint.py` measures ONE server and trusts its `usage` block. This
measures TWO against each other, and cannot trust `usage`: GenieAPIService
reports it as all zeros, and putting one arm on the usage block and the other
on a fallback would be two instruments rather than one measurement. So tokens
are counted locally from the returned text, with the bundle's own
tokenizer.json, identically for every arm.

Three things here are not incidental:

* **Interleaved passes.** A/B/B/A/A/B rather than all-A-then-all-B. The
  Hexagon is single-flight, so an arm's server must be stopped before the
  other starts, and a box that drifts over a twenty-minute run would otherwise
  hand the whole drift to whichever arm ran second. Interleaved, drift lands on
  both and shows up as spread instead of as a difference.

* **Decode as a two-request delta.** Same prompt at max_tokens=1 and
  max_tokens=1+N, subtracted. Prefill is identical in both and cancels, taking
  per-request HTTP overhead with it -- which is the only reason two different
  HTTP stacks are comparable. A sample is DISCARDED unless the long run really
  produced N more tokens; an early stop makes the subtraction meaningless and
  averaging it in is how a wrong number gets published.

* **Boundary-safe depths.** Genie picks the smallest compiled graph that fits
  at prefill time, so a depth where prompt + generated straddles a boundary
  runs the two calls on DIFFERENT graphs and the subtraction stops cancelling.
  The resulting noise reads convincingly as thermal decay. Keep
  depth + tokens inside one compiled length.

Not every server can be an arm. One that honours no output cap cannot have the
delta formed against it at all -- measured on GenieAPIService v2.3.7, which
returns 125 tokens for a requested 16 under both `max_tokens` and
`max_completion_tokens`.

Needs the NPU, a bundle, and `pip install tokenizers`. Unlike tests/, this is a
hardware tool.

The --geniex-model default names a model that must be IMPORTED first; standing
both servers up on one bundle is four steps for one of them and one command for
the other, all written down in "Reproducing the cross-server comparison" in
docs/GENIE_SERVER.md. Do that before wondering why an arm will not start.
"""
import argparse
import json
import os
import socket
import statistics
import subprocess
import sys
import time
import urllib.request

BUNDLE_DIR = os.environ.get("GENIE_BUNDLE_DIR", "")
SDK_DIR = os.environ.get("GENIE_SDK_DIR", "")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GENIEX = os.environ.get(
    "GENIEX_EXE",
    os.path.join(os.environ.get("LOCALAPPDATA", ""), "GenieX CLI", "geniex.exe"))


def _tokenizer():
    try:
        from tokenizers import Tokenizer
    except ImportError:
        sys.exit("this tool needs `pip install tokenizers` -- it counts tokens "
                 "locally because one of the servers under test reports usage "
                 "as all zeros")
    if not BUNDLE_DIR:
        sys.exit("set GENIE_BUNDLE_DIR to the bundle both servers will serve")
    return Tokenizer.from_file(os.path.join(BUNDLE_DIR, "tokenizer.json"))


TOK = None


def ntok(text):
    return len(TOK.encode(text, add_special_tokens=False).ids)


def prompt_at(depth):
    """A prompt of exactly `depth` tokens, measured rather than estimated."""
    unit = ("The measurement below concerns memory bandwidth on a mobile "
            "accelerator, and this paragraph repeats to reach a target depth. ")
    body = unit * max(1, depth // max(1, ntok(unit)))
    while ntok(body) < depth:
        body += unit
    return TOK.decode(TOK.encode(body, add_special_tokens=False).ids[:depth])


def _run(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)


def stop_all(arms):
    """Free the Hexagon. Single-flight means no two arms may hold it."""
    _run("taskkill /IM geniex.exe /F /T")
    for a in arms.values():
        _run('powershell -NoProfile -Command "$c=Get-NetTCPConnection '
             '-LocalPort %d -State Listen -ErrorAction SilentlyContinue; '
             'if($c){Stop-Process -Id $c.OwningProcess -Force}"' % a["port"])
    time.sleep(4)


def wait_port(port, secs=240):
    end = time.time() + secs
    while time.time() < end:
        try:
            socket.create_connection(("127.0.0.1", port), timeout=1).close()
            return True
        except OSError:
            time.sleep(2)
    return False


def start(name, arm):
    if name == "ours":
        subprocess.Popen(
            [sys.executable, os.path.join("src", "genie_server.py")], cwd=REPO,
            env={**os.environ, "GENIE_PORT": str(arm["port"])},
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        subprocess.Popen(
            [GENIEX, "serve", "--host", "127.0.0.1:%d" % arm["port"],
             "--keepalive", "3600"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return wait_port(arm["port"])


def chat(arm, prompt, cap, timeout):
    body = {"model": arm["model"],
            "messages": [{"role": "user", "content": prompt}],
            arm["cap"]: cap, "stream": False}
    req = urllib.request.Request(
        "http://127.0.0.1:%d/v1/chat/completions" % arm["port"],
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    wall = time.perf_counter() - t0
    m = d["choices"][0].get("message") or {}
    text = m.get("content") or ""
    # reasoning may arrive in a side channel; it is still decoded work
    for k in ("reasoning_content", "reasoning"):
        if isinstance(m.get(k), str):
            text += m[k]
    return wall, ntok(text)


def sample(arm, depth, n, timeout):
    p = prompt_at(depth)
    w1, t1 = chat(arm, p, 1, timeout)
    w2, t2 = chat(arm, p, 1 + n, timeout)
    steps, secs = t2 - t1, w2 - w1
    if steps < n * 0.9:
        return None, "early stop (%d of %d steps)" % (steps, n)
    if secs <= 0:
        return None, "non-positive delta -- prompt caching?"
    return steps / secs, "%d/%.2fs" % (steps, secs)


def clock_pct():
    r = _run('powershell -NoProfile -Command "$p=Get-CimInstance '
             'Win32_Processor;[math]::Round(100*$p.CurrentClockSpeed/'
             '$p.MaxClockSpeed,0)"')
    try:
        return int(r.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return -1


def main():
    global TOK
    ap = argparse.ArgumentParser()
    ap.add_argument("--depths", default="250,1500,3000",
                    help="keep depth + --tokens inside ONE compiled length")
    ap.add_argument("--tokens", type=int, default=120)
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--ours-port", type=int, default=8123)
    ap.add_argument("--geniex-port", type=int, default=18181)
    ap.add_argument("--ours-model", default="qwen3-4b-npu")
    ap.add_argument("--geniex-model", default="qualcomm/qwen3-4b-ours")
    ap.add_argument("--out", default="sweep-results.json")
    a = ap.parse_args()

    TOK = _tokenizer()
    depths = [int(x) for x in a.depths.split(",")]
    arms = {
        # cap: which spelling each server actually honours. Measured, not
        # assumed -- geniex ignores the legacy `max_tokens` outright.
        "ours": {"port": a.ours_port, "model": a.ours_model,
                 "cap": "max_tokens"},
        "geniex": {"port": a.geniex_port, "model": a.geniex_model,
                   "cap": "max_completion_tokens"},
    }

    order = []
    for i in range(a.passes):
        order += ["ours", "geniex"] if i % 2 == 0 else ["geniex", "ours"]
    print("interleaved: %s" % " -> ".join(order), flush=True)

    rows = []
    for idx, name in enumerate(order):
        stop_all(arms)
        print("\n[pass %d/%d] %s (clock %d%% of base)"
              % (idx + 1, len(order), name, clock_pct()), flush=True)
        if not start(name, arms[name]):
            print("   FAILED to start -- skipping", flush=True)
            continue
        try:
            chat(arms[name], prompt_at(64), 16, a.timeout)   # warmup, discarded
        except Exception as e:
            print("   warmup failed: %s" % str(e)[:120], flush=True)
        for d in depths:
            try:
                rate, note = sample(arms[name], d, a.tokens, a.timeout)
            except Exception as e:
                rate, note = None, "%s %s" % (type(e).__name__, str(e)[:90])
            print("   d%-5d %s (%s)"
                  % (d, ("%6.2f t/s" % rate) if rate else "  SKIP  ", note),
                  flush=True)
            if rate:
                rows.append({"arm": name, "pass": idx + 1, "depth": d,
                             "rate": round(rate, 3), "clock": clock_pct()})
    stop_all(arms)

    print("\n=== medians, full range in brackets ===", flush=True)
    for d in depths:
        out = []
        for name in arms:
            got = [r["rate"] for r in rows if r["arm"] == name and r["depth"] == d]
            out.append("%-7s %6.2f [%.2f-%.2f n=%d]"
                       % (name, statistics.median(got), min(got), max(got),
                          len(got)) if got else "%-7s no data" % name)
        print("d%-5d  %s" % (d, "   ".join(out)), flush=True)
    # Overlapping ranges mean the arms are indistinguishable at that depth.
    # Say so rather than quoting a ratio of two medians as though it were one.
    for d in depths:
        o = [r["rate"] for r in rows if r["arm"] == "ours" and r["depth"] == d]
        g = [r["rate"] for r in rows if r["arm"] == "geniex" and r["depth"] == d]
        if o and g:
            overlap = not (min(o) > max(g) or min(g) > max(o))
            print("d%-5d ranges %s" % (d, "OVERLAP -- indistinguishable"
                                       if overlap else "are disjoint"), flush=True)
    if a.out:
        json.dump({"rows": rows, "depths": depths, "tokens": a.tokens},
                  open(a.out, "w"), indent=1)
        print("\nwrote %s (%d samples)" % (a.out, len(rows)), flush=True)


if __name__ == "__main__":
    main()
