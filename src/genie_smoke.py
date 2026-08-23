#!/usr/bin/env python3
"""Smoke test for the Genie NPU server. Stdlib only. Exercises /v1/models,
non-streaming and streaming /v1/chat/completions, and prints timing."""
import json
import sys
import time
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080"


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=120) as r:
        return json.load(r)


def post(path, body, stream=False):
    data = json.dumps(body).encode()
    req = urllib.request.Request(BASE + path, data=data,
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=300)


print("== GET /v1/models =="); print(get("/v1/models"))

msgs = [{"role": "user", "content": "What is gravity? Answer in one short sentence."}]

print("\n== non-streaming ==")
t0 = time.time()
r = post("/v1/chat/completions", {"model": "qwen3-4b-npu", "messages": msgs, "max_tokens": 200})
obj = json.load(r)
dt = time.time() - t0
print("content:", obj["choices"][0]["message"]["content"][:400])
print("finish:", obj["choices"][0]["finish_reason"], "| wall %.1fs" % dt)

print("\n== streaming ==")
t0 = time.time(); first = None; n = 0; buf = []
r = post("/v1/chat/completions", {"model": "qwen3-4b-npu", "messages": msgs,
                                  "max_tokens": 200, "stream": True})
for raw in r:
    line = raw.decode("utf-8", "replace").strip()
    if not line.startswith("data:"):
        continue
    payload = line[5:].strip()
    if payload == "[DONE]":
        break
    d = json.loads(payload)["choices"][0]["delta"]
    if "content" in d:
        if first is None:
            first = time.time() - t0
        n += 1; buf.append(d["content"])
print("streamed:", "".join(buf)[:400])
print("chunks:", n, "| TTFT %.2fs" % (first or 0), "| wall %.1fs" % (time.time() - t0))
print("\nOK")
