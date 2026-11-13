from __future__ import annotations

import argparse
import json
import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from algorithms.base import OSFLAlgorithm
from datasets import domainnet, officehome, pacs
from datasets.clients import collect_client_records, configured_client_ids
from datasets.utils import ClipBatchCollator, PILImageRecords

PAYLOAD_VERSION = 3


def output_algorithm_name(granularity: str) -> str:
    """Return the stable artifact prefix for a FedDoG representation mode."""
    if granularity == "domain":
        return "feddog"
    if granularity == "class":
        return "feddog_c"
    raise ValueError(f"invalid granularity: {granularity!r}")


class _EmptyVaeFp32WarningFilter(logging.Filter):
    """Hide Diffusers' spurious AutoencoderKL warning with an empty module list."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not (
            record.name == "diffusers.models.modeling_utils"
            and message.startswith("There are modules in AutoencoderKL that should be kept in float32: []")
        )


logging.getLogger("diffusers.models.modeling_utils").addFilter(_EmptyVaeFp32WarningFilter())


def adaptive_conditioning_scale(rho: float, gamma: float) -> float:
    """Map MRL concentration to IP-Adapter strength."""
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    value = min(max(float(rho), 0.0), 1.0)
    numerator = value**gamma
    denominator = numerator + (1.0 - value) ** gamma
    return numerator / denominator if denominator > 0 else 0.0


def resolve_device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def model_dtype(name: str, device: torch.device) -> torch.dtype:
    choices = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = choices[name]
    if device.type == "cpu" and dtype == torch.float16:
        return torch.float32
    return dtype


def resolve_model_defaults(args: argparse.Namespace) -> None:
    """Resolve model paths that must change together for SD1.5 and SDXL."""
    if args.generation_batch_size is None:
        args.generation_batch_size = 4
    if args.guidance_scale is None:
        args.guidance_scale = 7.5
    artifact_prefix = output_algorithm_name(args.granularity)
    if args.sd_version == "sd15":
        args.sd_model = args.sd_model or "stable-diffusion-v1-5/stable-diffusion-v1-5"
        args.image_encoder_subfolder = args.image_encoder_subfolder or "models/image_encoder"
        args.ip_adapter_subfolder = args.ip_adapter_subfolder or "models"
        args.ip_adapter_weight = args.ip_adapter_weight or "ip-adapter_sd15.bin"
        default_representation_name = (
            f"{artifact_prefix}_{args.dataset}_clients_{args.clients_per_domain}_representations.pt"
        )
    else:
        args.sd_model = args.sd_model or "stabilityai/stable-diffusion-xl-base-1.0"
        args.image_encoder_subfolder = args.image_encoder_subfolder or "sdxl_models/image_encoder"
        args.ip_adapter_subfolder = args.ip_adapter_subfolder or "sdxl_models"
        args.ip_adapter_weight = args.ip_adapter_weight or "ip-adapter_sdxl.bin"
        default_representation_name = (
            f"{artifact_prefix}_{args.dataset}_sdxl_clients_{args.clients_per_domain}_representations.pt"
        )
    if args.representations_path is None:
        args.representations_path = str(
            Path(__file__).resolve().parents[1] / "outputs" / "models" / default_representation_name
        )
    if args.generated_data_dir is None:
        args.generated_data_dir = str(Path(__file__).resolve().parents[1] / "generated_data" / args.dataset)


def validate_payload(payload: Mapping[str, Any]) -> None:
    required = {
        "format", "version", "granularity", "num_classes", "domains", "class_names",
        "client_order", "client_domains", "clients",
    }
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"representation payload is missing keys: {sorted(missing)}")
    supported_formats = {"FedDoG-representations", "FedDoG-DomainNet-representations"}
    if payload["format"] not in supported_formats or payload["version"] != PAYLOAD_VERSION:
        raise ValueError("unsupported FedDoG representation payload")
    if payload.get("concentration_metric", "mrl") != "mrl":
        raise ValueError("representation payload uses a removed concentration metric; rerun extract")
    if payload["granularity"] not in {"domain", "class"}:
        raise ValueError(f"invalid granularity: {payload['granularity']!r}")
    if len(payload["class_names"]) != payload["num_classes"]:
        raise ValueError("class_names length does not equal num_classes")


def build_generation_conditions(
    payload: Mapping[str, Any],
    gamma: float,
    *,
    disable_daac: bool = False,
    fixed_ip_adapter_scale: float = 1.0,
) -> dict[int, list[dict[str, Any]]]:
    """Build per-client conditions for OSFL multi-domain generation."""
    if not math.isfinite(fixed_ip_adapter_scale) or not 0 <= fixed_ip_adapter_scale <= 1:
        raise ValueError("fixed_ip_adapter_scale must be between 0 and 1")
    clients = payload["clients"]
    clients_order = payload["client_order"]
    num_classes = int(payload["num_classes"])
    granularity = payload["granularity"]
    result: dict[int, list[dict[str, Any]]] = {}

    for class_id in range(num_classes):
        conditions = []
        for domain_index, client_name in enumerate(clients_order):
            source = clients[client_name]
            if granularity == "class":
                source = source["classes"].get(class_id)
                if source is None:
                    continue
            rho = float(source["rho"])
            conditions.append(
                {
                    "name": client_name,
                    "domain": payload["client_domains"][client_name],
                    "domain_index": domain_index,
                    "feature": source["feature"],
                    "rho": rho,
                    "scale": fixed_ip_adapter_scale if disable_daac else adaptive_conditioning_scale(rho, gamma),
                }
            )
        result[class_id] = conditions
    return result


def ip_adapter_image_embeds(
    feature: torch.Tensor,
    device: torch.device,
    do_classifier_free_guidance: bool,
    batch_size: int,
) -> list[torch.Tensor]:
    """Prepare raw image embeddings for diffusers' standard IP-Adapter API.

    One domain representation represents one virtual reference image, hence the
    shape is [batch, 1, projection_dim].  The loaded ImageProjection module is
    responsible for mapping it to the adapter's four cross-attention tokens.
    """
    conditional = F.normalize(feature.float(), dim=0).reshape(1, 1, -1).repeat(batch_size, 1, 1).to(device)
    if do_classifier_free_guidance:
        conditional = torch.cat([torch.zeros_like(conditional), conditional], dim=0)
    return [conditional]


class ClipFilter:
    """Frozen CLIP correctness and predictive-entropy filter (paper Section 3.7)."""

    def __init__(
        self,
        model_path: str,
        prompts: Sequence[str],
        device: torch.device,
        dtype: torch.dtype,
        entropy_threshold: float,
        normalize_entropy: bool,
        top_k: int = 1,
    ) -> None:
        from transformers import CLIPModel, CLIPProcessor

        self.device = device
        self.dtype = dtype
        self.entropy_threshold = entropy_threshold
        self.normalize_entropy = normalize_entropy
        if top_k not in {1, 5}:
            raise ValueError("top_k must be 1 or 5")
        self.top_k = top_k
        self.processor = CLIPProcessor.from_pretrained(model_path, backend="pil")
        self.model = CLIPModel.from_pretrained(
            model_path,
            torch_dtype=dtype,
        ).to(device)
        self.model.eval()
        self.model.requires_grad_(False)
        with torch.inference_mode():
            text_inputs = self.processor(text=list(prompts), padding=True, return_tensors="pt")
            text_inputs = {name: value.to(device) for name, value in text_inputs.items()}
            features = self.model.get_text_features(**text_inputs)
            if hasattr(features, "pooler_output"):
                features = features.pooler_output
            self.text_features = F.normalize(features.float(), dim=-1)

    @torch.inference_mode()
    def evaluate_batch(self, images: Sequence[Image.Image], target: int) -> list[dict[str, Any]]:
        if not images:
            return []
        image_inputs = self.processor(images=list(images), return_tensors="pt")
        pixel_values = image_inputs["pixel_values"].to(device=self.device, dtype=self.dtype)
        features = self.model.get_image_features(pixel_values=pixel_values)
        if hasattr(features, "pooler_output"):
            features = features.pooler_output
        features = F.normalize(features.float(), dim=-1)
        logit_scale = self.model.logit_scale.detach().float().exp()
        logits = (features @ self.text_features.T) * logit_scale
        probabilities = logits.softmax(dim=-1)
        predicted = logits.argmax(dim=-1)
        target_rank = (logits > logits[:, target : target + 1]).sum(dim=-1) + 1
        top_k = min(self.top_k, probabilities.shape[-1])
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)
        if self.normalize_entropy and probabilities.shape[-1] > 1:
            entropy /= math.log(probabilities.shape[-1])
        return [
            {
                "accepted": int(target_rank[index]) <= top_k and float(entropy[index]) <= self.entropy_threshold,
                "predicted": int(predicted[index]),
                "target_rank": int(target_rank[index]),
                "entropy": float(entropy[index]),
                "target_probability": float(probabilities[index, target]),
            }
            for index in range(len(images))
        ]


def existing_indices(class_dir: Path, prefix: str) -> set[int]:
    result: set[int] = set()
    for suffix in ("jpg", "jpeg", "png"):
        for image_path in class_dir.glob(f"{prefix}*.{suffix}"):
            tail = image_path.stem.removeprefix(prefix)
            if tail.isdigit():
                result.add(int(tail))
    return result


def append_json_line(path: Path, record: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def generated_output_dir(args: argparse.Namespace, gamma: float) -> Path:
    base = Path(args.generated_data_dir).expanduser().resolve()
    resolution_suffix = "_1024" if args.sd_version == "sdxl" else ""
    artifact_prefix = output_algorithm_name(args.granularity)
    conditioning = (
        f"fixed_scale_{getattr(args, 'fixed_ip_adapter_scale', 1.0):g}"
        if getattr(args, "disable_daac", False) else f"gamma_{gamma:g}"
    )
    filtering = (
        "no_filter" if getattr(args, "disable_filter", False)
        else f"top{args.filter_top_k}_entropy_{args.entropy_threshold:g}"
    )
    run_name = (
        f"{artifact_prefix}_{args.sd_version}{resolution_suffix}_{conditioning}_{filtering}_"
        f"clients_{args.clients_per_domain}_"
        f"images_{args.images_per_class}_seed_{args.seed}"
    )
    return base / run_name


def validate_args(args: argparse.Namespace) -> None:
    positive = {
        "num_classes": args.num_classes,
        "local_batch_size": args.local_batch_size,
        "images_per_class": args.images_per_class,
        "generation_batch_size": args.generation_batch_size,
        "max_attempts": args.max_attempts,
        "server_classifier_epochs": args.server_classifier_epochs,
        "server_classifier_batch_size": args.server_classifier_batch_size,
        "server_classifier_test_batch_size": args.server_classifier_test_batch_size,
    }
    invalid = [name for name, value in positive.items() if value < 1]
    if invalid:
        raise ValueError("these arguments must be positive: " + ", ".join(invalid))
    if args.local_num_workers < 0 or args.num_workers < 0:
        raise ValueError("worker counts cannot be negative")
    if args.samples_per_class is not None and args.samples_per_class < 1:
        raise ValueError("--samples-per-class must be positive")
    if args.entropy_threshold < 0:
        raise ValueError("--entropy-threshold cannot be negative")
    if args.gamma is not None and args.gamma <= 0:
        raise ValueError("--gamma must be positive")
    fixed_scale = getattr(args, "fixed_ip_adapter_scale", 1.0)
    if not math.isfinite(fixed_scale) or not 0 <= fixed_scale <= 1:
        raise ValueError("--fixed-ip-adapter-scale must be between 0 and 1")
    if "{}" not in args.prompt_template:
        raise ValueError("--prompt-template must contain '{}'")


class FedDoG(OSFLAlgorithm):
    """FedDoG/FedDoG-C client extraction, server generation and classifier training."""

    def __init__(self) -> None:
        self.args: argparse.Namespace
        self.payload: Mapping[str, Any] | None = None
        self.artifacts: dict[str, Any] = {}

    def load_payload(self) -> Mapping[str, Any]:
        if self.payload is None:
            representation_path = Path(self.args.representations_path).expanduser().resolve()
            if not representation_path.is_file():
                raise FileNotFoundError(
                    f"representation payload not found: {representation_path}; " "run with --stage extract first"
                )
            self.payload = torch.load(representation_path, map_location="cpu", weights_only=False)
            validate_payload(self.payload)
            if self.payload["granularity"] != self.args.granularity:
                raise ValueError(
                    f"--granularity={self.args.granularity} does not match representation payload "
                    f"granularity={self.payload['granularity']}"
                )
            if list(self.payload["domains"]) != self.args.domains.split(","):
                raise ValueError("representation payload domains do not match the configured dataset")
            if list(self.payload["client_order"]) != configured_client_ids(self.args):
                raise ValueError("representation payload clients do not match --clients-per-domain")
        return self.payload

    def default_synthetic_root(self, payload: Mapping[str, Any]) -> Path:
        gamma = self.args.gamma if self.args.gamma is not None else float(payload.get("gamma", 2.0))
        return generated_output_dir(self.args, gamma)

    @torch.inference_mode()
    def extract(self) -> None:
        """Simulate the one-round client uploads and save the server payload."""
        if self.args.dataset == "DomainNet":
            domains = domainnet.parse_domains(self.args.domains)
        elif self.args.dataset == "PACS":
            domains = pacs.parse_domains(self.args.domains)
        elif self.args.dataset == "OfficeHome":
            domains = officehome.parse_domains(self.args.domains)
        else:
            raise ValueError(f"unsupported dataset: {self.args.dataset}")
        output_path = Path(self.args.representations_path).expanduser().resolve()
        if output_path.is_file():
            payload = torch.load(output_path, map_location="cpu", weights_only=False)
            validate_payload(payload)
            matches_current_run = (
                str(payload.get("dataset", "")).lower() == self.args.dataset.lower()
                and payload.get("granularity") == self.args.granularity
                and payload.get("sd_version") == self.args.sd_version
                and int(payload.get("num_classes", -1)) == self.args.num_classes
                and list(payload.get("domains", [])) == domains
                and list(payload.get("client_order", [])) == configured_client_ids(self.args)
                and list(payload.get("class_names", [])) == list(self.args.categories)
                and int(payload.get("shots_per_class", -1)) == self.args.shots_per_class
                and int(payload.get("client_image_size", -1)) == self.args.image_size
                and payload.get("clip_model") == str(self.args.clip_model)
                and payload.get("image_encoder") == str(self.args.image_encoder)
                and payload.get("image_encoder_subfolder") == self.args.image_encoder_subfolder
                and payload.get("extraction_seed", getattr(self.args, "seed", 0)) == getattr(self.args, "seed", 0)
            )
            if matches_current_run:
                self.payload = payload
                self.artifacts["representations"] = str(output_path)
                print(f"Reusing existing representations: {output_path}")
                return
            print(f"Existing representations do not match the current settings; rebuilding: {output_path}")

        from transformers import CLIPProcessor, CLIPVisionModelWithProjection

        device = resolve_device(self.args.device)
        dtype = model_dtype(self.args.dtype, device)
        records_by_client, client_domains = collect_client_records(self.args)
        clients_order = list(records_by_client)
        canonical_names = dict(enumerate(self.args.categories))

        print(f"Loading frozen IP-Adapter image encoder from {self.args.image_encoder}")
        processor = CLIPProcessor.from_pretrained(self.args.clip_model, backend="pil")
        clip_model = CLIPVisionModelWithProjection.from_pretrained(
            self.args.image_encoder,
            subfolder=self.args.image_encoder_subfolder,
            torch_dtype=dtype,
        ).to(device)
        clip_model.eval()
        clip_model.requires_grad_(False)

        domain_payloads: dict[str, Any] = {}
        for client_name in clients_order:
            domain = client_domains[client_name]
            records = records_by_client[client_name]
            loader = DataLoader(
                PILImageRecords(records),
                batch_size=self.args.local_batch_size,
                shuffle=False,
                num_workers=self.args.local_num_workers,
                collate_fn=ClipBatchCollator(processor, self.args.image_size),
                pin_memory=device.type == "cuda",
                drop_last=False,
            )

            domain_sum: torch.Tensor | None = None
            domain_count = 0
            class_sums: dict[int, torch.Tensor] = {}
            class_counts: defaultdict[int, int] = defaultdict(int)
            print(f"[{client_name}] domain={domain}: extracting CLIP features from {len(records)} images")

            processed = 0
            for batch_index, (pixel_values, labels) in enumerate(loader, start=1):
                pixel_values = pixel_values.to(
                    device=device,
                    dtype=dtype,
                    non_blocking=device.type == "cuda",
                )
                features = clip_model(pixel_values=pixel_values, interpolate_pos_encoding=True).image_embeds
                features = F.normalize(features.float(), dim=-1).cpu()

                batch_sum = features.sum(dim=0)
                domain_sum = batch_sum if domain_sum is None else domain_sum + batch_sum
                domain_count += features.shape[0]
                if self.args.granularity == "class":
                    for feature, label in zip(features, labels.tolist()):
                        if label in class_sums:
                            class_sums[label] += feature
                        else:
                            class_sums[label] = feature.clone()
                        class_counts[label] += 1

                processed += features.shape[0]
                if batch_index % self.args.log_every == 0 or processed == len(records):
                    print(f"[{client_name}] {processed}/{len(records)}")

            if domain_sum is None or domain_count == 0:
                raise RuntimeError(f"client {client_name} produced no features")
            domain_mean = domain_sum / float(domain_count)
            domain_entry: dict[str, Any] = {"count": domain_count, "domain": domain}
            if self.args.granularity == "domain":
                domain_entry.update(
                    {
                        "feature": F.normalize(domain_sum, dim=0),
                        "rho": float(domain_mean.norm(p=2)),
                    }
                )
            else:
                class_entries: dict[int, Any] = {}
                for label in range(self.args.num_classes):
                    count = class_counts[label]
                    if count == 0:
                        continue
                    feature_sum = class_sums[label]
                    class_entries[label] = {
                        "feature": F.normalize(feature_sum, dim=0),
                        "rho": float((feature_sum / float(count)).norm(p=2)),
                        "count": count,
                    }
                domain_entry["classes"] = class_entries
            domain_payloads[client_name] = domain_entry

        # ``--stage all`` continues with SD/IP-Adapter in the same process. Release
        # the client encoder first so its memory is available to the server model.
        clip_model.to("cpu")
        del clip_model, processor
        if device.type == "cuda":
            torch.cuda.empty_cache()

        gamma = self.args.gamma if self.args.gamma is not None else (1.0 if self.args.granularity == "class" else 2.0)
        payload = {
            "format": "FedDoG-representations",
            "version": PAYLOAD_VERSION,
            "dataset": self.args.dataset,
            "granularity": self.args.granularity,
            "sd_version": self.args.sd_version,
            "gamma": gamma,
            "num_classes": self.args.num_classes,
            "domains": domains,
            "client_order": clients_order,
            "client_domains": client_domains,
            "class_names": [canonical_names[label] for label in range(self.args.num_classes)],
            "clip_model": str(self.args.clip_model),
            "image_encoder": str(self.args.image_encoder),
            "image_encoder_subfolder": self.args.image_encoder_subfolder,
            "shots_per_class": self.args.shots_per_class,
            "client_image_size": self.args.image_size,
            "clients": domain_payloads,
            "extraction_seed": self.args.seed,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, output_path)
        print(f"Saved compact client payload to {output_path}")
        for domain, entry in domain_payloads.items():
            if self.args.granularity == "domain":
                print(
                    f"  {domain:10s}: count={entry['count']:5d}, rho={entry['rho']:.4f}, "
                    f"lambda={adaptive_conditioning_scale(entry['rho'], gamma):.4f}"
                )
            else:
                class_entries = list(entry["classes"].values())
                mean_rho = sum(item["rho"] for item in class_entries) / len(class_entries)
                mean_scale = sum(adaptive_conditioning_scale(item["rho"], gamma) for item in class_entries) / len(
                    class_entries
                )
                print(
                    f"  {domain:10s}: count={entry['count']:5d}, classes={len(class_entries):3d}, "
                    f"mean_rho={mean_rho:.4f}, mean_lambda={mean_scale:.4f}"
                )
        self.payload = payload
        self.artifacts["representations"] = str(output_path)

    @torch.inference_mode()
    def generate(self) -> dict[str, Any]:
        from diffusers import (StableDiffusionPipeline,
                               StableDiffusionXLPipeline)

        payload = self.payload
        representation_path = Path(self.args.representations_path).expanduser().resolve()
        if payload is None:
            if not representation_path.is_file():
                raise FileNotFoundError(
                    f"representation payload not found: {representation_path}; run with --stage extract first"
                )
            payload = torch.load(representation_path, map_location="cpu", weights_only=False)
        validate_payload(payload)
        self.payload = payload
        if payload["granularity"] != self.args.granularity:
            raise ValueError(
                f"--granularity={self.args.granularity} does not match representation payload "
                f"granularity={payload['granularity']}"
            )
        payload_dataset = str(payload.get("dataset", "DomainNet"))
        if payload_dataset.lower() != self.args.dataset.lower():
            raise ValueError(
                f"--dataset={self.args.dataset} does not match representation payload dataset={payload_dataset}"
            )
        if self.args.num_classes != int(payload["num_classes"]):
            raise ValueError(
                f"configured num_classes={self.args.num_classes} does not match "
                f"payload num_classes={payload['num_classes']}"
            )
        payload_sd_version = str(payload.get("sd_version", "sd15"))
        if self.args.sd_version != payload_sd_version:
            raise ValueError(
                f"--sd-version={self.args.sd_version} does not match representation payload sd_version={payload_sd_version}; "
                f"run the extract stage again with --sd-version {self.args.sd_version}"
            )

        device = resolve_device(self.args.device)
        dtype = model_dtype(self.args.dtype, device)
        gamma = self.args.gamma if self.args.gamma is not None else float(payload.get("gamma", 2.0))
        class_names = list(payload["class_names"])
        prompts = [self.args.prompt_template.format(name.replace("_", " ")) for name in class_names]
        conditions = build_generation_conditions(
            payload, gamma,
            disable_daac=getattr(self.args, "disable_daac", False),
            fixed_ip_adapter_scale=getattr(self.args, "fixed_ip_adapter_scale", 1.0),
        )
        output_root = generated_output_dir(self.args, gamma)
        output_root.mkdir(parents=True, exist_ok=True)
        self.artifacts["synthetic_data"] = str(output_root)

        total_conditions = sum(len(entries) for entries in conditions.values())
        empty_classes = [class_id for class_id, entries in conditions.items() if not entries]
        if empty_classes:
            raise ValueError(f"no generation conditions available for classes: {empty_classes}")
        total_requested = total_conditions * self.args.images_per_class
        print(
            f"FedDoG generation plan: classes={len(class_names)}, conditions={total_conditions}, "
            f"images_per_class_per_client={self.args.images_per_class}, requested_images={total_requested}, "
            f"gamma={gamma:g}, "
            f"daac_enabled={not getattr(self.args, 'disable_daac', False)}, "
            f"fixed_ip_adapter_scale={getattr(self.args, 'fixed_ip_adapter_scale', 1.0):g}, "
            f"filter_enabled={not self.args.disable_filter}, "
            f"filter_top_k={self.args.filter_top_k}, entropy_threshold={self.args.entropy_threshold:g}, "
            f"generation_batch_size={self.args.generation_batch_size}, "
            f"output_dir={output_root}"
        )
        if self.args.dry_run:
            return {
                "requested": total_requested,
                "generated": 0,
                "existing": 0,
                "accepted": 0,
                "fallback": 0,
            }

        print(f"Loading frozen {self.args.sd_version.upper()} pipeline from {self.args.sd_model}")
        pipeline_class = StableDiffusionXLPipeline if self.args.sd_version == "sdxl" else StableDiffusionPipeline
        pipeline_load_options: dict[str, Any] = {"torch_dtype": dtype}
        if self.args.sd_version == "sdxl":
            pipeline_load_options["use_safetensors"] = True
            if dtype == torch.float16:
                pipeline_load_options["variant"] = "fp16"
        else:
            pipeline_load_options.update(safety_checker=None, requires_safety_checker=False)
        pipe = pipeline_class.from_pretrained(self.args.sd_model, **pipeline_load_options)
        pipe.load_ip_adapter(
            self.args.ip_adapter,
            subfolder=self.args.ip_adapter_subfolder,
            weight_name=self.args.ip_adapter_weight,
            image_encoder_folder=None,
        )
        pipe.set_progress_bar_config(disable=self.args.disable_diffusion_progress)
        if self.args.cpu_offload and device.type == "cuda":
            device_index = device.index if device.index is not None else torch.cuda.current_device()
            pipe.enable_model_cpu_offload(gpu_id=device_index)
        else:
            pipe.to(device)

        clip_filter: ClipFilter | None = None
        if not self.args.disable_filter:
            print(f"Loading frozen CLIP filter from {self.args.clip_model}")
            clip_filter = ClipFilter(
                model_path=self.args.clip_model,
                prompts=prompts,
                device=device,
                dtype=dtype,
                entropy_threshold=self.args.entropy_threshold,
                normalize_entropy=self.args.normalize_entropy,
                top_k=self.args.filter_top_k,
            )

        metadata_path = output_root / "generation_metadata.jsonl"
        summary = {
            "requested": total_requested,
            "generated": 0,
            "existing": 0,
            "accepted": 0,
            "fallback": 0,
        }
        do_cfg = self.args.guidance_scale > 1.0
        domains = list(payload["domains"])
        size_options = {"height": 1024, "width": 1024} if self.args.sd_version == "sdxl" else {}
        logged_domains: set[str] = set()

        for class_id, class_conditions in conditions.items():
            class_name = class_names[class_id]
            class_dir = output_root / prompts[class_id]
            class_dir.mkdir(parents=True, exist_ok=True)
            for condition in class_conditions:
                condition_target = self.args.images_per_class
                condition_name = str(condition["name"])
                domain_index = int(condition["domain_index"])
                prefix = f"train_{condition_name if domain_index >= 0 else 'global'}_"
                present = existing_indices(class_dir, prefix)
                pipe.set_ip_adapter_scale(float(condition["scale"]))
                pending = [index for index in range(condition_target) if index not in present]
                summary["existing"] += condition_target - len(pending)
                failed_attempts = 0
                best_reject: tuple[tuple[int, float, float], Image.Image, dict[str, Any]] | None = None
                if pending and condition_name not in logged_domains:
                    print(
                        f"[generate] domain={condition_name:10s} "
                        f"rho={float(condition['rho']):.4f} "
                        f"scale={float(condition['scale']):.4f}"
                    )
                    logged_domains.add(condition_name)

                while pending:
                    batch_size = self.args.generation_batch_size
                    embeds = ip_adapter_image_embeds(condition["feature"], device, do_cfg, batch_size)
                    batch_prompt = [prompts[class_id]] * batch_size
                    batch_negative_prompt = (
                        None if self.args.negative_prompt is None else [self.args.negative_prompt] * batch_size
                    )
                    images = pipe(
                        prompt=batch_prompt,
                        negative_prompt=batch_negative_prompt,
                        num_inference_steps=self.args.num_inference_steps,
                        guidance_scale=self.args.guidance_scale,
                        ip_adapter_image_embeds=embeds,
                        num_images_per_prompt=1,
                        **size_options,
                    ).images
                    if len(images) != batch_size:
                        raise RuntimeError(
                            f"diffusion returned {len(images)} images for a candidate batch of {batch_size}"
                        )

                    evaluations = (
                        [
                            {
                                "accepted": True,
                                "predicted": class_id,
                                "target_rank": 1,
                                "entropy": None,
                                "target_probability": None,
                            }
                            for _ in images
                        ]
                        if clip_filter is None
                        else clip_filter.evaluate_batch(images, class_id)
                    )
                    if len(evaluations) != len(images):
                        raise RuntimeError(f"CLIP returned {len(evaluations)} evaluations for {len(images)} images")

                    for image, evaluation in zip(images, evaluations):
                        if not pending:
                            break
                        image_index = pending[0]
                        attempt = failed_attempts + 1
                        record = {
                            "class_id": class_id,
                            "class_name": class_name,
                            "condition": condition_name,
                            "domain_index": domain_index,
                            "rho": float(condition["rho"]),
                            "ip_adapter_scale": float(condition["scale"]),
                            "image_index": image_index,
                            "attempt": attempt,
                            **evaluation,
                        }
                        if evaluation["accepted"]:
                            image_path = class_dir / f"{prefix}{image_index}.jpg"
                            image.save(image_path, quality=95)
                            record.update(
                                {
                                    "path": str(image_path.relative_to(output_root)),
                                    "saved": True,
                                    "fallback": False,
                                    "selection": "filter_accept" if clip_filter is not None else "unfiltered",
                                }
                            )
                            append_json_line(metadata_path, record)
                            summary["generated"] += 1
                            summary["accepted"] += 1
                            pending.pop(0)
                            print(
                                f"[accept] class={class_id:03d} domain={condition_name:10s} "
                                f"image={image_index:02d} attempt={attempt:02d} "
                                f"rank={evaluation['target_rank']} entropy={evaluation['entropy']}"
                            )
                            failed_attempts = 0
                            best_reject = None
                            continue

                        failed_attempts += 1
                        # Prefer a target within the configured top-k even if its entropy is too high.
                        # Otherwise prefer greater target probability, then lower entropy.
                        reject_rank = (
                            int(evaluation["target_rank"] <= self.args.filter_top_k),
                            float(evaluation["target_probability"]),
                            -float(evaluation["entropy"]),
                        )
                        if best_reject is None or reject_rank > best_reject[0]:
                            best_reject = (reject_rank, image, record.copy())
                        if self.args.log_rejections:
                            append_json_line(metadata_path, {**record, "saved": False, "fallback": False})
                            print(
                                f"[reject] class={class_id:03d} domain={condition_name:10s} "
                                f"image={image_index:02d} attempt={attempt:02d} "
                                f"target={class_id} pred={evaluation['predicted']} rank={evaluation['target_rank']} "
                                f"entropy={evaluation['entropy']:.4f}"
                            )
                        if failed_attempts < self.args.max_attempts:
                            continue

                        if best_reject is None:
                            raise RuntimeError("candidate rejection state was not recorded")
                        _, fallback_image, fallback_record = best_reject
                        image_path = class_dir / f"{prefix}{image_index}.jpg"
                        fallback_image.save(image_path, quality=95)
                        fallback_record.update(
                            {
                                "path": str(image_path.relative_to(output_root)),
                                "saved": True,
                                "fallback": True,
                                "selection": "best_reject",
                                "attempts_exhausted": self.args.max_attempts,
                            }
                        )
                        append_json_line(metadata_path, fallback_record)
                        summary["generated"] += 1
                        summary["fallback"] += 1
                        pending.pop(0)
                        print(
                            f"[fallback] class={class_id:03d} domain={condition_name:10s} "
                            f"image={image_index:02d} best_attempt={fallback_record['attempt']:02d} "
                            f"pred={fallback_record['predicted']} rank={fallback_record['target_rank']} "
                            f"entropy={fallback_record['entropy']:.4f} "
                            f"after={self.args.max_attempts} attempts"
                        )
                        failed_attempts = 0
                        best_reject = None

        summary_path = output_root / "generation_summary.json"
        summary_payload = {
            **summary,
            "algorithm": output_algorithm_name(str(payload["granularity"])),
            "granularity": payload["granularity"],
            "sd_version": self.args.sd_version,
            "resolution": 1024 if self.args.sd_version == "sdxl" else 512,
            "generation_mode": "multi-domain",
            "gamma": gamma,
            "daac_enabled": not getattr(self.args, "disable_daac", False),
            "fixed_ip_adapter_scale": (
                getattr(self.args, "fixed_ip_adapter_scale", 1.0)
                if getattr(self.args, "disable_daac", False) else None
            ),
            "filter_enabled": not self.args.disable_filter,
            "filter_top_k": None if self.args.disable_filter else self.args.filter_top_k,
            "entropy_threshold": None if self.args.disable_filter else self.args.entropy_threshold,
            "normalize_entropy": self.args.normalize_entropy,
            "generation_batch_size": self.args.generation_batch_size,
            "prompt_template": self.args.prompt_template,
            "images_per_class": self.args.images_per_class,
            "domains": domains,
            "clients": list(payload["client_order"]),
            "num_classes": len(class_names),
        }
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary_payload, handle, ensure_ascii=False, indent=2)
        print(f"Generation complete: {summary}")
        return summary

    def run(self, args: argparse.Namespace) -> dict[str, Any]:
        self.args = args
        self.args.output_algorithm_name = output_algorithm_name(self.args.granularity)
        self.payload = None
        self.artifacts = {}
        resolve_model_defaults(self.args)
        validate_args(self.args)

        if self.args.stage in {"extract", "all"}:
            self.extract()
        if self.args.stage in {"generate", "all"}:
            self.generate()
        if self.args.stage in {"train", "all"} and not self.args.dry_run:
            self.artifacts.update(self.train_global_classifier())
        return self.artifacts
