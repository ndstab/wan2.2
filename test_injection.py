import argparse
import logging
import os
import sys
import types

import torch

import wan
from wan.configs import SIZE_CONFIGS, WAN_CONFIGS


def _init_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(stream=sys.stdout)],
    )


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Sanity checks for reference video injection plumbing.")
    parser.add_argument(
        "--task",
        type=str,
        default="t2v-A14B",
        choices=list(WAN_CONFIGS.keys()),
        help="Task to test. For T2V use 't2v-A14B' or a compatible config.",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        required=True,
        help="Checkpoint directory used for normal inference.",
    )
    parser.add_argument(
        "--size",
        type=str,
        default="1280*720",
        choices=list(SIZE_CONFIGS.keys()),
        help="Output size key (must be supported by the chosen task).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A cloudy sky over the ocean, cinematic, realistic.",
        help="Text prompt for generation.",
    )
    parser.add_argument(
        "--ref_video",
        type=str,
        default=None,
        help="Path to reference video for injection tests.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="CUDA device index.",
    )
    parser.add_argument(
        "--sampling_steps",
        type=int,
        default=None,
        help="Override sampling steps (otherwise use config default).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    return parser.parse_args()


def _build_t2v(args):
    cfg = WAN_CONFIGS[args.task]
    if args.sampling_steps is not None:
        cfg.sample_steps = args.sampling_steps

    logging.info(f"Using config for task {args.task}: {cfg}")
    t2v = wan.WanT2V(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=args.device,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=False,
        convert_model_dtype=False,
    )
    return t2v, cfg


def _wrap_cross_attention_for_logging(model):
    """
    Wrap the first cross-attention block to log ref token and combined context shapes
    for Test 2 without changing core model logic.
    """
    if not hasattr(model, "blocks") or len(model.blocks) == 0:
        logging.warning("Model has no blocks; skipping cross-attention logging wrapper.")
        return

    block = model.blocks[0]
    if not hasattr(block, "cross_attn"):
        logging.warning("Block has no cross_attn; skipping wrapper.")
        return

    cross_attn = block.cross_attn
    orig_forward = cross_attn.forward

    def wrapped_forward(self, x, context, context_lens, ref_context=None, ref_context_lens=None):
        if ref_context is not None:
            logging.info(f"[Test 2] ref_tokens shape before model: {tuple(ref_context.shape)}")
            combined_seq_len = context.size(1) + ref_context.size(1)
            logging.info(f"[Test 2] combined context seq_len (text + ref): {combined_seq_len}")
        return orig_forward(
            x,
            context,
            context_lens,
            ref_context=ref_context,
            ref_context_lens=ref_context_lens,
        )

    block.cross_attn.forward = types.MethodType(wrapped_forward, cross_attn)


def _run_generation(t2v, cfg, args, ref_video_path=None, label="baseline"):
    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    video = t2v.generate(
        input_prompt=args.prompt,
        size=SIZE_CONFIGS[args.size],
        frame_num=cfg.frame_num,
        shift=cfg.sample_shift,
        sample_solver=cfg.sample_solver,
        sampling_steps=cfg.sample_steps,
        guide_scale=cfg.sample_guide_scale,
        seed=args.seed,
        offload_model=False,
        ref_video_path=ref_video_path,
    )

    torch.cuda.synchronize()
    peak_bytes = torch.cuda.max_memory_allocated(device)
    peak_gb = peak_bytes / (1024 ** 3)
    logging.info(f"[{label}] Output video shape: {tuple(video.shape)}")
    logging.info(f"[{label}] Peak GPU memory: {peak_gb:.2f} GB")
    return peak_gb


def main():
    _init_logging()
    args = _parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run the injection tests.")

    if args.task != "t2v-A14B":
        logging.warning(
            "This test script is tailored for T2V-style tasks; continuing with the chosen task anyway."
        )

    if args.ref_video is None:
        logging.warning(
            "No --ref_video provided. Test 2 and 3 will be skipped; only baseline will be run."
        )

    logging.info("Building WanT2V pipeline for tests...")
    t2v, cfg = _build_t2v(args)

    # Test 1 — Baseline unchanged
    logging.info("Running Test 1 — baseline generation without reference video...")
    baseline_peak = _run_generation(t2v, cfg, args, ref_video_path=None, label="Test 1 (no ref_video)")

    if args.ref_video is not None:
        # Wrap cross-attention to log shapes for Test 2
        logging.info("Wrapping cross-attention for Test 2 logging...")
        _wrap_cross_attention_for_logging(t2v.low_noise_model)

        # Test 2 — Injection plumbing
        logging.info("Running Test 2 — generation with reference video injection...")
        inj_peak = _run_generation(
            t2v,
            cfg,
            args,
            ref_video_path=args.ref_video,
            label="Test 2 (with ref_video)",
        )

        # Test 3 — OOM / memory delta check
        logging.info("Running Test 3 — memory overhead check...")
        delta_gb = inj_peak - baseline_peak
        logging.info(f"[Test 3] Additional GPU memory from injection: {delta_gb:.2f} GB")
        if delta_gb > 5.0:
            logging.warning(
                "Reference injection appears to add more than 5 GB over baseline. "
                "Please double-check compression steps and reference token handling."
            )
        else:
            logging.info(
                "Reference injection memory overhead is within the expected range (<= ~5 GB)."
            )


if __name__ == "__main__":
    main()

