# CLAUDE.md — Adverse Weather Video Synthesis on Wan2.2

## Meta: How to use this file

**Every session, every Claude instance, every new chat: read this file end-to-end before suggesting code changes or proposing experiments.** It is the source of truth for project state. If the user opens a chat without explicitly referencing this file, ask whether to load it before doing anything substantive.

When work progresses in a session, you (Claude) update this file:

1. **Tick checkboxes** (`- [ ]` → `- [x]`) as steps complete. Sub-steps under the active task get ticked as they happen.
2. **Append a dated entry to the Changelog** every time something material changes (new commit, new operating point, new metric value, parked experiment, etc.). The Changelog is **append-only** — never edit or delete prior entries, even if they turned out wrong. Wrong-then-corrected entries are valuable.
3. **Update Current State** if the active branch, last validated commit, or operating point changes.
4. **Move completed Tier tasks**: when a roadmap item is fully done, tick it; pick the next unticked task; copy that task's text into the **Active Task** section, replacing whatever was there.
5. **If a step's plan changes mid-execution**, edit the step text in place and note the change + reason in the Changelog.
6. **Conflict resolution**: if state in this file disagrees with the repo (different branch, different file content, missing file), trust the repo and update this file to match. Add a Changelog entry noting the discrepancy.

Do not reformat or restructure this file without being asked. Sections may be added, but the existing structure should be preserved so the user can find things in the same place across sessions.

---

## Project goal

Generate adverse weather videos on Wan2.2 TI2V 5B by injecting a reference weather video as style guidance, conditioned on a text prompt for scene content. Training-free (professor's hard constraint; lightweight adaptation may be discussable later).

**Input:** text prompt + reference weather video.
**Output:** video whose scene/motion follows the prompt and whose atmospheric appearance matches the reference.

Examples of target conditions: fog, heavy rain, sandstorm, snowstorm, volcanic ash, smoke, forest fire, low-light + fog. Practical motivation: data generation for autonomous-system perception under conditions where real footage is scarce.

---

## Hard constraints

- **No training loops, no `loss.backward()`, no optimizer steps.** Training-free only.
- **No modification of VAE, T5 encoder, or pretrained DiT weights.**
- **No pushing weight files** (`.pt`, `.pth`, `.safetensors`) to git.
- **Compute nodes have no internet** — all model downloads happen on the login node, cached under `~/.cache/`.

---

## Current state

*(Update this section whenever any of these change.)*

- **Active branch:** `latent_inspection` (forked off `self_att`, the project trunk). T1.1 scaffold validated end-to-end on the cluster 2026-08-14; both atlases produced. Uncommitted working-tree changes as of 2026-08-15: `analyze_captures.py` (new), `plot_analysis.py` (new), `decode_latent.py` (subset filters), `wan/textimage2video.py` (non-fatal capture-dump guard).
- **Last validated commit:** `ad78e0a` — autocast fix on top of the T1.1 capture scaffold. (`6a4dd4d` remains the last commit validated for *injection behaviour*.)
- **Cluster storage constraint (2026-08-14):** the `ai-at-ieor` group quota on `/scratch` is **exhausted (3T/3T)**. Large writes fail with `Disk quota exceeded` after a few tens of MB. Captures therefore live on **node-local `/tmp`** and die with the job allocation. Needs a quota increase from the PI/admins — this is a project-wide blocker, not specific to this task.
- **Last validated operating point (2026-08-16): `λ_ref=0.10`, all 30 blocks, all timesteps.** Prompt "car on city street", seed 42, sandstorm ref, 704×1280, 61 frames, 50 steps. Scene content (car, road, buildings, trees, lane markings) remains legible *and* the sandstorm is clearly applied — the first output meeting the project's stated goal.
- **Superseded operating point:** `λ_ref=0.5`, all blocks — total content destruction (output is a featureless dust field, no scene at all). Every result recorded before 2026-08-16 used this setting, i.e. ~5× past the usable range. The historical "content collapse" was an operating-point error, not a flaw in the injection pathway.
- **λ curve (7 points, videos in `outputs/lambda-sweep/`):** 0.00 no weather → 0.05 content intact / weather mild → **0.10 best tradeoff** → 0.15 content degrading → 0.25–0.50 content lost. Degradation is progressive, not a cliff. The orange bottom-edge banding artifact scales with λ and is essentially absent at ≤0.10.
- **Working style metric:** CSD (Contrastive Style Descriptors, ViT-L). Replaces the earlier WSIS. PickStyle's reported CSD = 0.37 on a related task is the external benchmark.
- **Content metric:** TBD. Probably LPIPS against a no-injection baseline run with the same seed + prompt.

---

## Architecture quick-reference

**Model:** Wan2.2 TI2V 5B. 30 DiT blocks. `dim=3072`, `num_layers=30`, `num_heads=24`. Each block: Self-Attn → Cross-Attn → FFN. 3D VAE with temporal stride 4, spatial stride 16 (`vae_stride = (4, 16, 16)` in `wan/configs/wan_ti2v_5B.py`; TI2V 5B uses 16, unlike the 14B-series VAE which uses 8).

**Reference injection (current, post-`6a4dd4d`):**

- Reference video → same 3D VAE as content latent → same `patch_embedding` → same RoPE grid as content. Zero-padded to match `seq_len`.
- At each denoising step, ref is re-noised to the current σ via flow-matching interp: `x_t = (1-σ)·x_0 + σ·noise`, `σ = t/T`. Keeps ref tokens statistically consistent with content at every step.
- Inside `WanSelfAttention.forward`: Q from content only; K and V are per-token convex blends:
  ```python
  k = (1 - λ) * k_content + λ * k_ref
  v = (1 - λ) * v_content + λ * v_ref
  ```
- One attention pass — not noisy-self-attn then ref-self-attn. The K/V projection matrices are the same ones used for content (no new weights).
- AdaLN modulation from the timestep is applied to ref with the same parameters as content *before* self-attn.
- CFG: ref injected into both conditional and unconditional passes.
- `--inject_blocks` selects which of the 30 blocks receive the injection (defaults to all).

**Why this works where the previous cross-attn concatenation didn't:** the old cross-attn K/V projections were trained on T5 text embeddings; CLIP/VAE ref features were out-of-distribution there. The new self-attn K/V projections were trained on video latent tokens after patch-embed + adaLN — exactly the representation the ref now arrives in. Plus ref tokens share a positional grid with content tokens, enabling positionally coherent attention.

**Key files:**

- `wan/modules/model.py` — `WanSelfAttention`, `WanCrossAttention`, `WanAttentionBlock`, `WanModel`. Injection logic lives here.
- `wan/textimage2video.py` — `WanTI2V.t2v()` denoising loop. Re-noising of ref to current timestep happens here.
- `wan/distributed/sequence_parallel.py` — multi-GPU path. Mirror any `model.py` change here.
- `generate.py` — CLI entry. Args: `--ref_video`, `--lambda_ref`, `--inject_blocks`.
- `compute_csd.py` (formerly `compute_wsis.py`) — style metric.

**Cluster context:**

- IITB HPC. 160 GB GPU. 2 GPUs via `torchrun --nproc_per_node=2`.
- Checkpoint: `/scratch/IITB/ai-at-ieor/23b0702/Wan2.2/Wan2.2-TI2V-5B/`
- Repo: `/scratch/IITB/ai-at-ieor/23b0702/Wan2.2/`
- Ref videos: `…/ref_videos/`
- Outputs: `…/outputs/`
- Long-running jobs: tmux session + Jupyter dummy script holding the GPU allocation.

---

## Git / branch protocol

**Branches:**

- `main` — pristine fork of upstream Wan. Do not modify. Used for upstream pulls and diff baselines.
- `develop` — trunk of this project. All validated work lands here. (Promoted from `self-attention`; treat `self-attention` and `develop` as the same trunk until renamed.)
- `cfg` — CS-CFG implementation. Parked. Revisit only if Tier 1 produces a clear need.
- `exp/<descriptor>` — every new experimental direction. Examples: `exp/latent-inspection`, `exp/last-3-blocks`, `exp/v-only`, `exp/depth-scaled-lambda`, `exp/sidecar-attn`.

**Workflow per experiment:**

1. Branch off `develop`: `git checkout -b exp/<descriptor> develop`.
2. Implement the minimum needed. Commit experiment config to `experiments/<descriptor>.yaml` (prompt, seed, ref video path, λ, CLI invocation, expected output dir).
3. Run on cluster. Save outputs to `outputs/<descriptor>/<timestamp>/`. Never overwrite.
4. When a result is recorded, **tag**: `git tag result/YYYY-MM-DD-<descriptor>-<metric>-<value>`. Push the tag.
5. Decision:
   - **Validated** → merge into `develop` (squash preferred, keeps history readable).
   - **Did not validate** → leave the branch and tag in place. Do not delete. Failure cases are needed for the writeup.
6. Never force-push tags. Never overwrite output directories.

Combining two experimental directions: branch off the merged `develop`, then cherry-pick from the other `exp/` branch.

---

## Roadmap

### Tier 1 — deep-dive ablation (professor's directive)

The professor's meeting brief: do a deep dive into the architecture. Map which blocks do structure vs appearance, which timesteps do layout vs detail. Inspect intermediate latents directly. First specific thing to try: convex combination in only the last 2-3 blocks, possibly with a different attention module there.

- [ ] **T1.1 Intermediate-latent inspection scaffold.** Add hooks that dump latents at specified `(block_idx, timestep_idx)` pairs and decode them through the VAE to viewable frames. Build a baseline atlas (no injection, fixed prompt + seed) of "what does block N at timestep t produce normally." Re-run with injection, diff side-by-side. **Branch:** `exp/latent-inspection`.

- [ ] **T1.2 Block subset ablation.** Fix prompt ("car on city street"), ref (sandstorm), seed=42, λ=0.5. Sweep `--inject_blocks`: `[29]`, `[28,29]`, `[27,28,29]`, `[25-29]`, `[22-29]`, `[15-29]`, all. Record CSD, content-preservation proxy (LPIPS vs no-injection baseline), visual inspection. Start with last 2-3 (professor's first hypothesis). **Branch:** `exp/block-subset-sweep`.

- [ ] **T1.3 Timestep subset ablation.** Fix winning block subset from T1.2. Vary injection timesteps: last 10%, 25%, 50%, all. Same metrics. **Branch:** `exp/timestep-sweep`.

- [ ] **T1.4 (Conditional) Block × timestep cross product.** Only if T1.2 and T1.3 each show clean monotonic structure. Otherwise skip — multiplying noise gives more noise. **Branch:** `exp/block-timestep-grid`.

### Tier 2 — architectural refinements

Address the professor's point that a single λ is too coarse a knob; the content/structure-vs-style asymmetry should be expressed *where* and *what* you inject, not in a scalar.

- [ ] **T2.1 V-only injection.** Blend only V, keep K from content. Q-K selects where to attend purely from content (preserving structure), V supplies the style content. Cleanest expression of "structure from content, texture from ref." **Branch:** `exp/v-only`.

- [ ] **T2.2 Depth-scaled λ.** `λ(block_idx)` ramps up toward later blocks. Encodes the structure→texture prior architecturally. **Branch:** `exp/depth-scaled-lambda`.

- [ ] **T2.3 Timestep-scaled λ.** `λ(t)` ramps up toward end of denoising (when fine detail is being painted in). **Branch:** `exp/timestep-scaled-lambda`.

- [ ] **T2.4 Sidecar attention head in last 2-3 blocks.** Separate attention module: Q from content, K/V from ref, summed into the residual with a small fixed scale, alongside the normal self-attn. Probably what the professor meant by "different attention module" for last 2-3 blocks. Still training-free (init scale to small constant). **Branch:** `exp/sidecar-attn`.

- [ ] **T2.5 CS-CFG sanity check.** Hold a working Tier-1 operating point fixed; toggle CS-CFG on/off. If no improvement in CSD/content tradeoff, drop. CS-CFG adds ~50% inference cost (third forward pass). **Branch:** `cfg` (already exists).

### Tier 3 — deferred

- [ ] **T3.1** Temporal encoder swap (X-CLIP / InternVideo2) for ref encoding. Probably unnecessary given VAE-based ref pathway already captures temporal structure.
- [ ] **T3.2** Formal tolerance threshold τ (professor's earlier requirement). `WSIS/CSD > τ AND content_drift < δ` defines valid operating region. Plot tradeoff curve across λ sweep.
- [ ] **T3.3** 14B model extrapolation.
- [ ] **T3.4** Failure-mode catalog for the writeup: cherry-picked cases where injection breaks in instructive ways.

---

## Active task

*(Replace this section's contents when starting a new task. Copy the current Tier roadmap item here, expanded into sub-steps. Tick sub-steps as you go.)*

**Task:** T1.1 — Intermediate-latent inspection scaffold.

**Why this first:** Without it, every Tier-1 ablation is guessing from a single endpoint number. With it, you can point to where style enters and where content gets damaged across the (block, timestep) grid. This is the literal "deep dive into the architecture" the professor asked for.

**Sub-steps:**

- [x] Branch off develop: `git checkout -b exp/latent-inspection develop`. *(User created `latent_inspection` off `self_att`; equivalent.)*
- [x] Add a capture mechanism on `WanAttentionBlock.forward`. *(Implemented inline in `WanModel.forward` rather than via a separate `LatentCapture` hook class — keeps the change inside one file and avoids adding hook-management surface area. Captures post-block residual `x` (the full self-attn + cross-attn + FFN output), keyed by `(block_idx, t_idx)`, fp16/CPU. Mirrored in `sp_dit_forward` with cross-rank gather.)*
- [x] CLI flags in `generate.py`: `--capture_blocks`, `--capture_timesteps`, `--capture_dir` (all three must be set together).
- [x] Save captured latents. **Design deviation:** single `captures.pt` dump per run (under `--capture_dir`) instead of one `.pt` per (block, t). Reason: atomic flush at end-of-loop is simpler, smaller filesystem load, and metadata (grid_sizes, t-values, patch_size) lives alongside the tensors. The dump is a dict with `{meta, grid_sizes, seq_len, timestep_values, blocks: {(block, t): tensor}}`.
- [x] Add `decode_latent.py`: loads `captures.pt`, re-derives `e` from saved `t` values, applies `model.head → unpatchify → vae.decode`, saves middle-frame PNG per `(block, t)` as `block_<NN>_t_<TT>.png`.
- [x] **Baseline run:** no injection, fixed prompt + seed. **Grid revised per TA (2026-08-14):** `--capture_blocks 0,5,10,15,20,25,29 --capture_timesteps 0,2,4,6,8,10,12,14,16,18,20,30,40,49` (TA asked for even steps 0–20; 30/40/49 added as late anchors since style lands late and extra capture points are free at generation time). **Frame count reduced 121 → 61** because `ti2v-5B` supports only `704*1280`/`1280*704` (no 480p variant), so frame count is the only cost lever. Done: `baseline-20260814_202759`, 98/98 cells, 8.48 GB, 9m24s, exit 0.
- [x] **Injected run:** λ=0.5, all blocks, same prompt + seed, same capture points. Done: `injected-20260814_204155`, sandstorm ref, 98/98 cells, 8.48 GB, ~9m30s, exit 0.
- [x] Diff: quantitative (block × timestep) heatmaps via new `analyze_captures.py` + `plot_analysis.py`. Observations in Changelog. Pixel-space decode of a 7×4 subset in progress.
- [ ] Tag: `result/YYYY-MM-DD-latent-inspection-baseline`. *(User action — Claude does not run git.)*

**How to run (cluster):**

*(Revised 2026-08-14. We now hold a **single** GPU per TA instruction, so the
2-GPU `torchrun --ulysses_size 2` form is retired — runs go through
`WanModel.forward`, not `sp_dit_forward`. Captures write to **node-local
`/tmp`** because the `ai-at-ieor` group quota on `/scratch` is exhausted.)*

Wrapper scripts on the cluster (outside the repo, in `~`):

- `~/run_atlas.sh <run_name> [ref_video] [lambda]` — one atlas arm.
- `~/run_analysis.sh` — block × timestep metrics over both dumps.
- `~/run_decode.sh` — subset decode of both arms to PNG.

All are launched detached so they survive an SSH drop:
```
screen -dmS atlas bash -lc "srun --jobid=<JOBID> --overlap ~/run_atlas.sh <name> [ref] [lambda] > <log> 2>&1"
```

Underlying invocation (baseline; add `--ref_video ... --lambda_ref 0.5` for the injected arm):
```
python generate.py --task ti2v-5B \
    --ckpt_dir /scratch/IITB/ai-at-ieor/23b0702/Wan2.2/Wan2.2-TI2V-5B \
    --size 1280*704 --frame_num 61 --sample_steps 50 \
    --offload_model True --convert_model_dtype --t5_cpu \
    --prompt "car on city street" --base_seed 42 \
    --capture_blocks 0,5,10,15,20,25,29 \
    --capture_timesteps 0,2,4,6,8,10,12,14,16,18,20,30,40,49 \
    --capture_dir /tmp/wan_atlas/<run_name> \
    --save_file /tmp/wan_atlas/<run_name>/video.mp4
```

Analysis (CPU; matplotlib is NOT installed in the cluster `torch` env, so plot locally):
```
python analyze_captures.py \
    --baseline /tmp/wan_atlas/<baseline>/captures.pt \
    --injected /tmp/wan_atlas/<injected>/captures.pt \
    --output_dir /tmp/wan_atlas/analysis --no_plots
# then copy the (tiny) CSVs off-cluster and:
python plot_analysis.py --analysis_dir outputs/latent-inspection/analysis
```

Decode a subset (single GPU, after a run):
```
python decode_latent.py \
    --captures /tmp/wan_atlas/<run>/captures.pt \
    --ckpt_dir /scratch/.../Wan2.2-TI2V-5B \
    --output_dir /tmp/wan_atlas/decoded/<run> \
    --only_blocks 0,5,10,15,20,25,29 --only_timesteps 0,10,20,49
```

---

## Quick experiment to run in parallel

Orthogonal to T1.1, runnable today: same prompt, ref, seed=42, λ=0.5, but `--inject_blocks 27,28,29` only. If last-3 gives a clearly better content/style tradeoff than all-blocks, the professor's first hypothesis is validated in one cluster job and T1.2 priority is confirmed. One job, one number.

---

## Open questions

- ~~Does the current code cheaply skip the ref forward pass for blocks excluded from `--inject_blocks`, or pay full cost regardless?~~ **Answered 2026-08-14 (code read + timing):** it skips cheaply. `ref_hidden_states` is patch-embedded once per forward pass; the per-block extra work (`norm1` + `Wk`/`Wv` on ref) only runs for `idx in inject_set` (`model.py:622`). Measured: injected run (all 30 blocks) ≈ 9m30s vs baseline 9m24s at 61 frames — injection overhead is ~1%, so block sweeps are cheap.
- Is CSD computed frame-wise then averaged, or temporally aware? Frame-averaged misses temporal weather dynamics (rain motion, smoke billowing).
- Is the all-blocks/λ=0.5 content collapse monotonic in λ? At λ=0.3 does content come back roughly? Cheap to check; logs the slope.
- Where does Wan2.2 keep its block-level CFG scale (if any)? Confirms whether sidecar-attn (T2.4) can be added cleanly.

---

## Changelog

*(Append-only. Format: `YYYY-MM-DD — branch/scope — what changed — outcome.`)*

- 2026-XX-XX — `self-attention` — Migrated from cross-attn concatenation to self-attn KV-mixing (commit `6a4dd4d`). VAE-based ref encoder, per-step re-noising, shared RoPE grid. Cross-attn restored to text-only. — Style transfer strong, content lost at λ=0.5 all-blocks. Confirms pathway works; lever too strong.
- 2026-XX-XX — `compute_csd.py` — Replaced WSIS (CLIP cosine) with CSD ViT-L. — Benchmark-comparable to PickStyle's reported 0.37.
- 2026-XX-XX — `cfg` — Implemented CS-CFG (three forward passes, independent style/content scales). — Parked pending Tier-1 results; effectiveness uncertain.
- 2026-05-13 — repo root — Created CLAUDE.md (this file). — Project state consolidated; Tier 1 active.
- 2026-05-16 — `latent_inspection` (off `self_att`) — Implemented T1.1 capture scaffold: `_init_captures` / `_capture_block_output` / `_dump_captures` on `WanModel`; mirrored in `sp_dit_forward` (with `gather_forward` across SP ranks); plumbed `--capture_blocks`, `--capture_timesteps`, `--capture_dir` through `generate.py` → `WanTI2V.generate()` → `t2v()`. Captures only fire on the conditional CFG pass. Added `decode_latent.py`. Added `*.pt`/`*.pth`/`*.safetensors`/`captures/` to `.gitignore`. — Smoke tests of helper logic pass locally; awaiting cluster run for baseline + injected atlases.
- 2026-05-16 — `latent_inspection` — Design deviation from roadmap: dump is one `captures.pt` per run instead of per-(block,t) `.pt` files. Decoded PNGs still follow the `block_<NN>_t_<TT>.png` naming. — Simpler atomic flush, less filesystem churn on parallel FS.
- 2026-05-17 — `latent_inspection` — Bug fix in `decode_latent.py`: `_make_e_for_step` was building the time embedding under an outer bf16 autocast, which made `time_embedding` (an `nn.Linear`) emit bf16. `Head.forward` then tripped its `assert e.dtype == torch.float32` and the bare `except` swallowed the empty error string. Wrapped the construction in `autocast('cuda', dtype=torch.float32)` to mirror `WanModel.forward`, and added `traceback.format_exc()` to the failure logger so future plumbing bugs surface visibly. — Decode now succeeds end-to-end.
- 2026-05-17 — `latent_inspection` — Smoke test cleared end-to-end on the HPC (single-GPU, 10 sample steps, 21 frames, 704×1280, prompt "car on city street", seed 42, *no ref*, capture_blocks=[0,29], capture_timesteps=[0,9]). Produced 4 PNGs decoded via `head + unpatchify + VAE.decode`. Confirms: capture mechanism fires correctly, dump+reload preserves shape/dtype, decode pipeline reaches pixel space. — Pipeline validated; ready for the actual baseline + injected atlas A/B with full sampling steps.
- 2026-06-01 — repo root — Added `ARCHITECTURE.md`: long-form teaching companion to this file. Covers the original Wan2.2 TI2V 5B inference pipeline (T5 → DiT block stack → Head → unpatchify → VAE.decode, plus the timestep×block nested loops), the modified flow with StyleID-style ref injection (VAE-encoded ref, per-step re-noising, shared RoPE grid, K/V blend inside self-attn, CFG-both-branches, `--inject_blocks` gating), why the prior cross-attn-concat path failed, and a shapes cheat-sheet for default 1280×704/121-frame inference (seq_len=27280, dim=3072, 24 heads × 128 head_dim, latent `[48, 31, 44, 80]`). ASCII diagrams throughout. — Onboarding doc for future sessions; CLAUDE.md remains the state-of-the-project reference.
- 2026-08-06 — meta — Correction to the prior entry: `ARCHITECTURE.md` was actually created **2026-08-06** (today), not 2026-06-01. Prior entry left in place per append-only rule. — Timeline note for future sessions.
- 2026-08-06 — meta — Mentor meeting prep session (user returning to the project after a long break; mentor also returning after a longer gap). No code changes this session. Activities: (a) authored `ARCHITECTURE.md`; (b) verbal walkthrough for the user of why we pivoted cross-attn → self-attn KV-mixing and how to explain it, and what the 4-PNG smoke test actually is / isn't. Repo state is unchanged since 2026-05-17: T1.1 scaffold implemented and smoke-tested; the full baseline + injected atlases (the actual scientific output of T1.1) are still pending on the cluster. — Mentor may redirect priorities in the meeting today; whoever picks this up next should ask the user for the meeting outcome before assuming the roadmap is unchanged.
- 2026-08-14 — meta / TA directions — Meeting outcome relayed by user. TA asked for: (a) more blocks — capture grid set to `[0,5,10,15,20,25,29]`; (b) timesteps `0–20`, even only; (c) "trend capturing — how representation changes across timesteps and across blocks"; (d) a *meta-system* research direction: physics-based / environmental style **estimation** from the reference ("estimation of things, no fixed equations"), operating indirectly/behind the scenes, where the DiT is untouched and only *extracted representations* are injected. Claude added late anchors `30,40,49` to the capture grid (style lands late; extra capture points cost ~nothing at generation time). — Roadmap updated; Tier-1 grid revised.
- 2026-08-14 — infra — **Group quota blocker.** First atlas attempt (`baseline-20260814_200854`) completed sampling in 10m26s then died in `torch.save` with `PytorchStreamWriter failed writing file data/19`. Diagnosis: `lfs quota -g ai-at-ieor /scratch` = **3T used / 3T limit**; a `dd` test returned `Disk quota exceeded` after 32 MB. OSTs only 4–5% full, so quota not space; user quota is unlimited. Fixes: (1) captures relocated to node-local `/tmp` on the GPU node (90 GB free); (2) `wan/textimage2video.py` now wraps `_dump_captures` in try/except so a storage failure can no longer destroy a completed generation (previously the exception propagated and killed VAE decode, losing the video too); (3) corrupt 826 MB partial `captures.pt` deleted (zip footer never written → unloadable). — Unblocked, but node-local captures die with the job allocation.
- 2026-08-14 — `latent_inspection` — **T1.1 COMPLETE.** Single GPU (A100 80GB, job 366309). Both arms at 704×1280 / 61 frames / 50 steps / seed 42 / prompt "car on city street". Baseline `baseline-20260814_202759` (9m24s) and injected `injected-20260814_204155` (sandstorm, λ=0.5, all 30 blocks, ~9m30s), each 98/98 cells, 8.48 GB. Added `analyze_captures.py` (cross-run divergence + within-run block delta, CPU, mmap) and `plot_analysis.py` (heatmaps, run off-cluster since the cluster env has no matplotlib). — **Findings:** (i) at step 0, where both runs share an identical initial latent and the measurement is therefore drift-free, injection effect peaks at **blocks 15–20** (cos_dist 0.079 at b15; rel_l2 0.57 at b20) and is **largely re-absorbed by block 29** (cos_dist 0.012, barely above block 0's 0.004). This cuts against the professor's "inject only in the last 2–3 blocks" hypothesis and argues for sweeping mid-depth subsets in T1.2. (ii) Token norms inflate **~2× (1.83–2.07)** at step 49 across blocks 5–29, versus 0.85–1.05 through steps 0–20 — a late-stage magnitude blow-up. (iii) Within-run, block 29 does the largest single-block transformation (rel_l2 ~1.03–1.11 vs 0.36–0.73 elsewhere) over *fewer* blocks of separation. (iv) Corroborating: baseline video 9.9 MB vs injected 2.7 MB H.264 at identical resolution/frames. **Caveat:** representational divergence ≠ style quality; these numbers say where internal state moves, not which movement is desirable. Cross-run cells at steps ≥2 also carry accumulated trajectory drift — only the step-0 column is confound-free. — Pixel-space decode of a 7×4 subset in progress to disambiguate.
- 2026-08-15 — `latent_inspection` — **CORRECTION to the 2026-08-14 T1.1 entry, finding (i).** That entry claimed injection is "largely re-absorbed by block 29" based on `cos_dist` 0.012 and `rel_l2` 0.188 at block 29 / step 0. Both metrics are *ratios* normalised by the baseline residual magnitude, and **block 29 roughly doubles that magnitude** (mean token norm 114.53 ± 0.32 vs 53–63 at blocks 5–25 — note the near-zero std: block 29 drives every token to almost the same length). Recomputing the *absolute* difference as `rel_l2 × ‖baseline‖_F`, with `‖·‖_F = sqrt(N·(mean² + std²))`: 498 (b0), 1631 (b5), 2613 (b10), **3869 (b15), 3882 (b20)**, 2702 (b25), **2556 (b29)**. So block 29 retains **~66% of the peak perturbation**, a ~34% reduction — not an erasure. **What survives:** the mid-depth peak at blocks 15–20 is real in absolute terms. **What does not:** the claim that block 29 absorbs the injection, and therefore the strength of the argument against the professor's "inject only into the last 2–3 blocks" hypothesis. Last-blocks injection remains a legitimate thing to test in T1.2 rather than something the atlas has ruled out. — Prior entry left in place per the append-only rule; lesson: relative metrics need their denominator inspected before any depth-wise claim.
- 2026-08-15 — literature scan (TA request: check CVPR/ECCV this year for collisions) — Weather-video generation is crowded but almost entirely **trained**: AutoAWG (ICMR '26, CogVideoX1.5-5B, fine-tuned, depth/lineart/sketch controls), AutoWeather4D (arXiv 2603.26546, WAN-FUN backbone, feed-forward but **user-set** physical parameters with explicit BRDF/radiative-transfer equations), WeatherWeaver (arXiv 2505.00704, trained, 6-dim weather vector), Cyclone (arXiv 2607.13927), SafeDrive (CVPR 2026 **Workshop**). Training-free style transfer is also crowded but generic/artistic and mostly images: FreeViS, UniVST, OmniTransfer, TeleStyle, StyleGallery, MAST, VISTA. — **Two consequences.** (1) **Collision:** `Scheduled Style Injection` (arXiv 2605.26538) is training-free, varies injection strength **across timesteps**, and frames it as "expanding the style-content Pareto frontier" — i.e. our T2.3 (and partly T2.2) is already published, though images-only on Stable Diffusion. Demote those from "novel contribution" to "cited technique." (2) **Open slot confirmed:** nobody found *estimates* weather parameters **from a reference video**, training-free, and injects the extracted representation — which is precisely the TA's meta-system direction. That should become the paper's core claim, with the block×timestep atlas as the supporting analysis contribution. ECCV 2026 official accept list not verified (searches returned arXiv, not proceedings).
- 2026-08-16 — `latent_inspection` — **λ sweep: working operating point found.** Motivated by the T1.1 videos, which showed λ=0.5 does not "over-style" but *replaces the scene with the reference's own appearance* (baseline = crisp street; injected = featureless dust field). Ran 5 extra generations (λ = 0.05, 0.10, 0.15, 0.25, 0.35), same prompt/seed/ref/frames, no captures, ~9.5 min each; combined with the existing λ=0 and λ=0.5 this gives a 7-point curve. **Result: degradation is progressive in λ, and λ≈0.10 is the operating point** — dust visibly obscures the skyline while the car, road, trees and lane markings stay legible. λ=0.05 keeps content nearly intact but the weather is mild; λ=0.15 is already degrading; ≥0.25 is destroyed. Also: the orange bottom-edge band scales with λ and is near-absent at ≤0.10, so it is a symptom of over-injection rather than an independent bug. **Two intermediate claims by Claude were wrong and are corrected here:** (a) H.264 file size was used as a first proxy and suggested a "cliff" — wrong, heavy haze compresses well regardless of whether structure survives; (b) with only 5 points, λ=0.15 was called "usable" — with 0.05/0.10 filled in, 0.10 is clearly better. — **Consequences:** T1.2's block-subset sweep should run at λ≈0.10, not 0.5 (at 0.5 every subset destroys content, so the sweep would have been uninformative); CSD/LPIPS should be measured at the new operating point; and the long-standing "content collapse" framing in this file is an operating-point artifact, not a property of self-attn KV-mixing.

---

## Reference papers

- **StyleID** (arXiv 2312.09008) — current self-attn KV-mixing pattern.
- **StyleCrafter** (arXiv 2312.00330) — dual cross-attn; superseded by self-attn pivot but useful for writeup framing of why concat-into-cross-attn fails.
- **PickStyle** (arXiv 2510.07546) — built on Wan2.1-VACE-14B; CS-CFG; CSD = 0.37 SOTA. Closest precedent for our setup.
- **InstantStyle** — block ablation methodology on SDXL. We apply this to Wan2.2 (claimed original contribution).
- **WeatherWeaver** — parametric weather decomposition; reference for evaluation framing.
- **VACE** — video conditioning unit; potential pivot target if Wan2.2 path stalls (would mean switching base model to Wan2.1-VACE).

---

## Things NOT to do

- No training loops, optimizers, or `loss.backward()` calls.
- Do not modify VAE, T5 encoder, or existing DiT weights.
- Do not concatenate reference tokens into the *sequence dimension* of self-attention. The current design blends K and V in place; concatenation along seq_len is a different (and rejected) approach.
- Do not push model weights to git.
- Do not delete experimental branches that didn't pan out — they're part of the writeup.
- Do not overwrite output directories. Always use a timestamped subfolder.
- **Claude must never autonomously create branches, commit, push, open PRs, merge PRs, tag, or perform any other git state-changing action.** All such actions are user-only decisions. Claude's role is strictly to *guide* the user — explain what command to run, why, and when — and wait for the user to execute it. This applies even when the Git / branch protocol above says "branch off `develop`," "tag the result," "merge into `develop`," etc.: those steps describe what *the user* will do, and Claude only proposes them. If Claude believes a git action is warranted, it must surface the recommendation and stop, not run it.