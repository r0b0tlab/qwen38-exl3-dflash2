# PR plan — DFlash2 draft support for turboderp-org/exllamav3

Prepared 2026-09-16. Source branch: `r0b0tlab/exllamav3` `dflash2-pathway` @ `5cdf6bc`
(9 commits, base `0740edc`; master is `02aef45`). Diff: 9 files, ~1254 insertions
(3 code files + 1 registration line + 4 test files).

## 0. Contributor guidelines in effect (verified against the repo)

There is no CONTRIBUTING.md, no PR template and no docs/style guide — the conventions are the
repo's de facto practice, confirmed from its history:

- **Commit style**: `Area: Summary` (e.g. "MoE: Fix gather for TP-split layers").
- **PR body structure** (from merged PR #349): `### Why` (deep technical motivation),
  `### Commits` (numbered walkthrough of every commit), plus accuracy/speed/testing sections.
  PR #349 also shows docs get updated in the same PR when user-facing.
- **Tests**: `tests/test_*.py`, pytest-collectable; most recent additions carry a trailing
  underscore (`test_moe_cpu_tiers_.py`, `test_bc_recurrent_graph_geometry_.py`, ...);
  resource-dependent tests are gated by env vars (`EXL3_TEST_MODEL`) and skip cleanly.
- **CI**: `.github/workflows/build.yml` is a `workflow_dispatch` wheel-build workflow — there
  is NO test CI on PRs; the maintainer reviews manually. The PR touches no build config.
- **Docs**: neither README nor `doc/` documents the v1 DFlash pathway, so a code+tests PR is
  consistent with the repo's current state (a brief README line is optional).
- **Provenance**: the math is ported from MIT `z-lab/dflash`; provenance comments already in
  the code. ExLlamaV3 is MIT — compatible.

## 1. Branch preparation (exact steps)

1. **Rebase**: `git fetch origin && git rebase origin/master`. The branch is exactly one
   commit behind (`02aef45` "Don't include torch/extension.h from CUDA"); zero file overlap
   with our diff, so a clean rebase is expected. Afterwards `git diff --stat master..branch`
   must show only our 9 files.
2. **Commit curation — fold the fixes into their logical parents.** Reviewers read this list;
   "fix the fix" chains within one PR read poorly. Target structure (5 commits, upstream
   voice):
   1. `arch: DFlash2DraftModel config + model skeleton` (`ab97ba7`)
   2. `modules: GroupedDynamicCausalConv + CandidateSelector (z-lab parity)` (`6778825`,
      `5b0617d`)
   3. `generator: propose wiring + q-aware rejection verify` (`d2eb017`)
   4. `dflash2: loaded-model dtype, walk, crop and batch fixes` (fold `cefec6c`, `b8e1a00`,
      `aa6adf8`, `4960047`, `5cdf6bc` — or better: squash each fix into the commit it fixes)
   5. `tests: DFlash2 config/modules/load/verify suites`
   (Preferred: interactive rebase folding each fix into its parent so the history reads as if
   written correctly once.)
3. **Test renames + gating**: `ntest_dflash2_*.py` → `tests/test_dflash2_{config,modules,load,
   verify}_.py` (trailing underscore per convention). Gate with a dedicated env var
   (`EXL3_TEST_DFLASH2`, following `EXL3_TEST_MODEL`) plus GPU presence; verify a bare
   `pytest tests/ -k dflash2` skips green on a machine without the draft checkpoint.
4. **One review-anticipated hardening**: when `model_init`/`Generator` attaches a draft model,
   assert the target cache reserves enough recurrent history (`max_history >= draft window`)
   — today a hand-rolled harness fails deep inside the first verify pass; fail loudly at
   setup instead. Small, generator-side, add a test.

## 2. Verification before posting

- `pytest tests/ -k dflash2` green with the checkpoint present; existing generator tests
  unaffected.
- Fresh e2e smoke on the rebased tree: draft conversion (`convert.py -b 4.00` on
  `incoai/Qwen3.8-27B-DFlash2`), `examples/chat.py -m <target> -dm <draft>` generation, and a
  one-question acceptance run — confirm the numbers below reproduce post-rebase.
- Diff self-review against every entry in §0.

## 3. PR description (structure per #349; full text: `notes/PR-DFLASH2.md`)

- **### Why** — DFlash2 extends the v1 pathway with checkpoint-carried modules (grouped
  dynamic conv around each sublayer + candidate path selector); measured +1.3 accepted
  tokens/step over MTP on a quantized 27B target on one consumer GPU.
- **### Commits** — numbered walkthrough (final 5-commit structure).
- **### Accuracy** — GSM8K greedy mean acceptance: **5.474** (DFlash2, EXL3 draft) vs
  5.463 (BF16 draft) vs 4.101 (MTP head) vs 1.000 (auto-regressive), RTX 3090, EXL3
  4.00 bpw target; multi-needle NIAH passes at 262,080 tokens.
- **### Speed** — 152.4 tok/s vs 113.9 (MTP) vs 41.8 (AR) at 8k ctx; batch ladder
  109.0 → 134.6 → 152.1 tok/s at 1/2/4 concurrent sequences (fp32 GDN verify states cap
  slots at ~4 on 24 GB).
- **### Testing** — exact commands; which tests need a checkpoint/GPU and how they gate.
- **### Notes for review** — (a) quantized target lm_head is deliberate and lossless
  w.r.t. the deployed head (p and q come from the same deterministic EXL3 head; SGLang
  rejects quantized heads, we don't need to); (b) the sampled verify covers plain
  temperature/top-k/top-p samplers only and falls back to token-match acceptance otherwise;
  (c) the selector is a no-op in the module walk — alternative implementation: a
  `fwd_end_idx`-style boundary as in `deepseek_v4_mtp`; (d) interplay with open PR #261
  (multi-draft): both touch `generator.py` but in orthogonal places — flagging for the
  maintainer's sequencing.
- **### Related** — SGLang PR 35371, vLLM PR 52816, llama.cpp PR 27342, `z-lab/dflash`
  (MIT reference for all parity tests).

## 4. Posting and process

- Open from `r0b0tlab/exllamav3:dflash2-pathway` → `turboderp-org/exllamav3:master`.
- Reference the results repo (https://github.com/r0b0tlab/qwen38-exl3-dflash2) and the
  container image in the body; keep container/scripts/notes out of the PR diff.
- Keep the branch rebased while under review; do not bundle unrelated experiments.
