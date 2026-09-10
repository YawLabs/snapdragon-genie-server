# Open-sourcing this repo

**Recommendation: publish the code and the findings. Decide the model bundle
separately, and do not let it block the rest.**

Status: proposed, 2026-09-09. Owner: Jeff.

---

## Why

The repo is already written as a public artifact. Its own README opens with
"a measurement log for running an LLM on the Snapdragon X Elite Hexagon NPU,
and a server that applies what it found", and says the findings "are the
valuable part and they are portable -- most are properties of Genie and of the
AI Hub bundles, not of this code, so they hold whichever server you run."

That is a distribution argument. Findings that hold whichever server you run
are worth nothing to anybody while they sit in a private repo, and the work of
making them publishable is already done.

Three further reasons, in order of weight:

1. **It is the evidence behind a marketing claim we are about to make.**
   typed.cloud is adding copy about a free tier that runs on the customer's own
   machine, and a blog post about local inference. "Runs on your machine" is a
   sentence; a public measurement log with the arithmetic shown is proof. We
   cannot link to a private repo, so today the strongest supporting asset we
   own is unusable.

2. **The credibility comes from the negative results, and we already have
   them.** The README states outright that the server "is **not** faster than
   Qualcomm's own server; that was measured, and decode is a tie", and
   elsewhere "So do not choose this for speed." On 2026-09-07 the repo
   self-audited and *deleted* eleven previously-published numbers it could not
   evidence. A vendor publishing a negative result about its own work, and
   retracting its own unsupported figures, is rare enough to be the reason the
   post travels. That posture cannot be manufactured later; it is already in
   the git history.

3. **The adjacent, more sensitive artifact is already public.**
   `YawLabs/llama.cpp` -- the fork carrying an experimental QNN backend -- is
   public today. Keeping the measurement log private while shipping the code
   is the wrong way round: the code without the findings is the half that
   invites misuse.

## What is already done

Verified 2026-09-09 against the working tree:

| Check | Result |
|---|---|
| Licence | Apache-2.0, present |
| `NOTICE` | present, with third-party attribution |
| Hardcoded personal paths (`C:\Users\<name>`, `/c/Users/<name>`) | **zero matches** across `.py`, `.md`, `.ps1`, `.json` |
| Secret-shaped strings | none; every hit is API vocabulary (`max_tokens`, `stop_sequences`, `tokens_in_flight`) |
| Repo size | 80 files, no large binaries tracked |
| Framing | README already addresses a public reader |

Nothing on that list is the usual reason a repo cannot be flipped. The work
that remains is editorial and legal, not hygiene.

## Before flipping the switch

1. **Decide the model bundle separately.**
   `publish/qwen3.5-9b-genie-npu/` carries a `publish.ps1`. Redistributing a
   *converted model bundle* raises the upstream model licence and the Qualcomm
   AI Hub bundle terms, which is a different question from open-sourcing our
   own code and our own measurements. Split it: publish the repo, and treat
   bundle redistribution as its own decision with its own licence review. If
   that decision is not ready, remove or gate `publish/` for the initial flip
   rather than delaying everything behind it.

2. **Make the n=1 caveat unmissable.**
   Every number in the repo is one Snapdragon X Elite X1E80100, Hexagon v73,
   Windows on ARM64, and in places n=1 to n=3. The docs say so repeatedly, but
   a reader arriving from an aggregator reads the headline number and not the
   methodology. Put the hardware and the sample size in the README's first
   screen, not only beside each table. A post or repo that generalises to
   "local inference" rather than "local inference on this class of machine" is
   making a claim this evidence does not support.

3. **Reconcile the figures that another repo has already cited.**
   typed's `docs/adr/019-local-multi-engine-routing.md` cites a `poll:true`
   result as 0.78x in roughly six places. A later controlled A/B here appears
   to reverse that. Two public repos disagreeing about the same measurement is
   the kind of thing a careful reader finds immediately, and it costs more
   credibility than either number is worth. Settle which is current and align
   both before they are both public.

4. **State the support posture in the README.**
   This is a research log and a reference implementation, not a supported
   product. Say so plainly and say what a bug report should contain, or the
   issue tracker becomes a support queue for hardware almost nobody has.

5. **Confirm the rename landed everywhere.**
   The remote is `YawLabs/snapdragon-genie-server`; the local checkout is still
   `snapdragon-npu-llm`, and at least one document in the typed repo linked the
   old name -- a URL that 404s for the rename and would have been private
   anyway. Grep both repos for the old spelling before publishing, because
   after the flip a dead link is a public dead link.

## Sequence

1. Resolve item 1 (bundle in or out of the initial publish).
2. Apply items 2 and 4 -- README edits, half an hour.
3. Resolve item 3 with whichever measurement is current; correct the loser.
4. Grep both repos for `snapdragon-npu-llm` and fix every hit.
5. Flip visibility to public.
6. Only then link it: from the typed blog post, and from `docs/LOCAL_MODEL.md`
   where the worked example currently has to describe the Genie server without
   being able to point at it.

Steps 1 to 4 are the whole cost. None of them requires new measurement.

## What not to publish

- Anything under a directory holding vendor SDKs or converted bundles, unless
  item 1 clears it. The `genie-npu/` sidecar is not part of this repo and
  should stay that way.
- Any figure that the 2026-09-07 self-audit removed. If a number cannot be
  reproduced from a script in the repo, it should not come back in the README
  on the way out of the door.
- Watt figures. Power draw is described as the NPU's real advantage and is
  currently unmeasured anywhere; publishing an estimate as though it were
  measured would undo exactly the credibility the negative results buy.

## Open questions for the owner

- Bundle redistribution: in scope for the first publish, or deferred?
- Licence stays Apache-2.0? It is already the declared licence, so this is a
  confirmation rather than a decision.
- Does the repo want issues enabled at all on day one, or discussions only?
