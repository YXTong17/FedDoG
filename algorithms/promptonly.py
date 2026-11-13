from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from algorithms.base import OSFLAlgorithm
from datasets.clients import configured_client_ids
def resolve_promptonly_defaults(args: argparse.Namespace) -> None:
    if args.generation_batch_size is None:
        args.generation_batch_size = 4
    args.guidance_scale = 7.5 if args.guidance_scale is None else args.guidance_scale
    if args.sd_version == "sd15":
        args.sd_model = args.sd_model or "stable-diffusion-v1-5/stable-diffusion-v1-5"
    else:
        args.sd_model = args.sd_model or "stabilityai/stable-diffusion-xl-base-1.0"
    if args.generated_data_dir is None:
        args.generated_data_dir = str(Path(__file__).resolve().parents[1] / "generated_data" / args.dataset)


def generated_output_dir(args: argparse.Namespace) -> Path:
    base = Path(args.generated_data_dir).expanduser().resolve()
    resolution = 1024 if args.sd_version == "sdxl" else 512
    run_name = (
        f"promptonly_{args.sd_version}_{resolution}_"
        f"clients_{args.clients_per_domain}_"
        f"images_{args.images_per_class}_seed_{args.seed}"
    )
    return base / run_name


def validate_args(args: argparse.Namespace) -> None:
    positive = {
        "images_per_class": args.images_per_class,
        "generation_batch_size": args.generation_batch_size,
        "num_inference_steps": args.num_inference_steps,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        raise ValueError("these Prompt Only arguments must be positive: " + ", ".join(invalid))
    if args.guidance_scale < 1:
        raise ValueError("--guidance-scale must be at least 1")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative")
    if args.samples_per_class is not None and args.samples_per_class <= 0:
        raise ValueError("--samples-per-class must be positive")
    if "{}" not in args.prompt_template:
        raise ValueError("--prompt-template must contain '{}'")


def build_metadata(args: argparse.Namespace) -> dict[str, Any]:
    domains = args.domains.split(",")
    clients = configured_client_ids(args)
    return {
        "format": "PromptOnly-metadata",
        "dataset": args.dataset,
        "domains": domains,
        "class_names": list(args.categories),
        "num_classes": args.num_classes,
        "generation_slots": clients,
        "sd_version": args.sd_version,
        "sd_model": args.sd_model,
    }


def generate_images(args: argparse.Namespace, metadata: Mapping[str, Any]) -> dict[str, Any]:
    output_root = generated_output_dir(args)
    class_names = list(metadata["class_names"])
    slots = list(metadata["generation_slots"])
    total = len(class_names) * len(slots) * args.images_per_class
    print(
        f"Prompt Only generation plan: classes={len(class_names)}, sample_groups={len(slots)}, "
        f"images_per_class_per_group={args.images_per_class}, images={total}, output_dir={output_root}"
    )
    if args.dry_run:
        return {"requested": total, "generated": 0, "existing": 0, "output_dir": str(output_root)}

    from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]
    if device.type == "cpu" and dtype != torch.float32:
        dtype = torch.float32
    pipeline_class = StableDiffusionXLPipeline if args.sd_version == "sdxl" else StableDiffusionPipeline
    load_options: dict[str, Any] = {"torch_dtype": dtype}
    if args.sd_version == "sdxl":
        load_options["use_safetensors"] = True
        if dtype == torch.float16:
            load_options["variant"] = "fp16"
    else:
        load_options.update(safety_checker=None, requires_safety_checker=False)
    pipe = pipeline_class.from_pretrained(args.sd_model, **load_options)
    pipe.set_progress_bar_config(disable=args.disable_diffusion_progress)
    if args.cpu_offload and device.type == "cuda":
        device_index = device.index if device.index is not None else torch.cuda.current_device()
        pipe.enable_model_cpu_offload(gpu_id=device_index)
    else:
        pipe.to(device)

    output_root.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"requested": total, "generated": 0, "existing": 0}
    size_options = {"height": 1024, "width": 1024} if args.sd_version == "sdxl" else {}
    for class_id, class_name in enumerate(class_names):
        prompt = args.prompt_template.format(class_name.replace("_", " "))
        class_dir = output_root / prompt
        class_dir.mkdir(parents=True, exist_ok=True)
        for slot_index, slot_name in enumerate(slots):
            prefix = f"train_{slot_name}_"
            pending = [
                index
                for index in range(args.images_per_class)
                if not (class_dir / f"{prefix}{index}.jpg").is_file()
            ]
            summary["existing"] += args.images_per_class - len(pending)
            while pending:
                indices = pending[: args.generation_batch_size]
                generator = torch.Generator(device=device).manual_seed(
                    args.seed + slot_index * 1_000_000 + class_id * 10_000 + indices[0]
                )
                images = pipe(
                    prompt=[prompt] * len(indices),
                    negative_prompt=(
                        None if args.negative_prompt is None else [args.negative_prompt] * len(indices)
                    ),
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    generator=generator,
                    num_images_per_prompt=1,
                    **size_options,
                ).images
                for image_index, image in zip(indices, images):
                    image.save(class_dir / f"{prefix}{image_index}.jpg", quality=95)
                    summary["generated"] += 1
                del pending[: len(indices)]
                print(
                    f"Prompt Only generated group={slot_name:18s} class={class_id:03d} "
                    f"images={','.join(str(index + 1) for index in indices)}/{args.images_per_class}"
                )

    summary.update(
        {
            "algorithm": "promptonly",
            "dataset": args.dataset,
            "domains": list(metadata["domains"]),
            "sample_groups": slots,
            "class_names": class_names,
            "sd_version": args.sd_version,
            "sd_model": args.sd_model,
            "resolution": 1024 if args.sd_version == "sdxl" else 512,
            "prompt_template": args.prompt_template,
            "negative_prompt": args.negative_prompt,
            "guidance_scale": args.guidance_scale,
            "num_inference_steps": args.num_inference_steps,
            "images_per_class": args.images_per_class,
            "seed": args.seed,
            "output_dir": str(output_root),
        }
    )
    with (output_root / "generation_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    del pipe
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


class PromptOnly(OSFLAlgorithm):
    """Server-only class-prompt image generation baseline."""

    def __init__(self) -> None:
        self.args: argparse.Namespace
        self.metadata: Mapping[str, Any] | None = None
        self.artifacts: dict[str, Any] = {}

    def load_payload(self) -> Mapping[str, Any]:
        if self.metadata is None:
            self.metadata = build_metadata(self.args)
        return self.metadata

    def default_synthetic_root(self, payload: Mapping[str, Any]) -> Path:
        del payload
        return generated_output_dir(self.args)

    def run(self, args: argparse.Namespace) -> dict[str, Any]:
        if args.stage == "extract":
            raise ValueError("Prompt Only has no client extraction stage; use --stage generate, train, or all")
        self.args = args
        self.artifacts = {}
        resolve_promptonly_defaults(args)
        validate_args(args)
        self.metadata = build_metadata(args)

        if args.stage in {"generate", "all"}:
            generation = generate_images(args, self.metadata)
            self.artifacts["synthetic_data"] = generation["output_dir"]
            self.artifacts["generation"] = generation
        if args.stage in {"train", "all"} and not args.dry_run:
            self.artifacts.update(self.train_global_classifier())
        return self.artifacts
