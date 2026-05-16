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

- **Active branch:** `latent_inspection` (forked off `self_att`, the project trunk). Work in progress — T1.1 scaffold landed but not yet validated against a cluster run.
- **Last validated commit:** `6a4dd4d` — StyleID-style KV-mixing in self-attention; reference re-noised via 3D VAE at every denoising step.
- **Last validated operating point:** all 30 blocks injected, all timesteps, `λ_ref=0.5`.
- **Behavior at that operating point:** heavy style transfer, content largely overwhelmed. The injection pathway is confirmed to be working; the lever is cranked too hard. This is the *expected* failure mode for uniform full-strength KV-mixing.
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
- [ ] **Baseline run:** no injection, fixed prompt + seed, capture default block/timestep set. Save under `outputs/latent-inspection/baseline-<timestamp>/`. **Suggested defaults:** `--capture_blocks 0,7,14,21,28,29 --capture_timesteps 0,12,25,37,49 --sample_steps 50`. Est. disk: ~4.8 GB per run (30 captures × 160 MB at default 1280×704, 121-frame, bf16). Expand to all 30 blocks only after this works.
- [ ] **Injected run:** λ=0.5, all blocks, same prompt + seed, same capture points. Save under `outputs/latent-inspection/injected-<timestamp>/`.
- [ ] Diff: side-by-side decoded frames per (block, timestep). Eyeball where injection visibly diverges from baseline. Note observations in Changelog.
- [ ] Tag: `result/YYYY-MM-DD-latent-inspection-baseline`.

**How to run (cluster):**

Baseline (no ref):
```
torchrun --nproc_per_node=2 generate.py \
    --task ti2v-5B --ckpt_dir /scratch/IITB/ai-at-ieor/23b0702/Wan2.2/Wan2.2-TI2V-5B \
    --size 1280*704 --frame_num 121 --sample_steps 50 \
    --prompt "car on city street" --base_seed 42 \
    --capture_blocks 0,7,14,21,28,29 --capture_timesteps 0,12,25,37,49 \
    --capture_dir outputs/latent-inspection/baseline-$(date +%Y%m%d_%H%M%S) \
    --ulysses_size 2
```

Injected (sandstorm ref, λ=0.5):
```
torchrun --nproc_per_node=2 generate.py \
    --task ti2v-5B --ckpt_dir /scratch/.../Wan2.2-TI2V-5B \
    --size 1280*704 --frame_num 121 --sample_steps 50 \
    --prompt "car on city street" --base_seed 42 \
    --ref_video /scratch/.../ref_videos/sandstorm.mp4 --lambda_ref 0.5 \
    --capture_blocks 0,7,14,21,28,29 --capture_timesteps 0,12,25,37,49 \
    --capture_dir outputs/latent-inspection/injected-$(date +%Y%m%d_%H%M%S) \
    --ulysses_size 2
```

Decode (single GPU, after a run):
```
python decode_latent.py \
    --captures outputs/latent-inspection/<run>/captures.pt \
    --ckpt_dir /scratch/.../Wan2.2-TI2V-5B \
    --output_dir outputs/latent-inspection/<run>/decoded
```

---

## Quick experiment to run in parallel

Orthogonal to T1.1, runnable today: same prompt, ref, seed=42, λ=0.5, but `--inject_blocks 27,28,29` only. If last-3 gives a clearly better content/style tradeoff than all-blocks, the professor's first hypothesis is validated in one cluster job and T1.2 priority is confirmed. One job, one number.

---

## Open questions

- Does the current code cheaply skip the ref forward pass for blocks excluded from `--inject_blocks`, or pay full cost regardless? Affects how expensive block sweeps will be.
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