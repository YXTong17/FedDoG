from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from algorithms.feddog import FedDoG
from algorithms.promptonly import PromptOnly
from datasets.domainnet import (
    DOMAINNET_CATEGORIES,
    DOMAINNET_DEFAULT_ROOT,
    DOMAINNET_DOMAINS,
    DOMAINNET_IMAGE_SIZE,
)
from datasets.officehome import (
    OFFICEHOME_CATEGORIES,
    OFFICEHOME_DEFAULT_ROOT,
    OFFICEHOME_DOMAINS,
    OFFICEHOME_IMAGE_SIZE,
)
from datasets.pacs import PACS_CATEGORIES, PACS_DEFAULT_ROOT, PACS_DOMAINS, PACS_IMAGE_SIZE

ROOT = Path(__file__).resolve().parent
DATASET_NAMES = ("DomainNet", "PACS", "OfficeHome")
ALGORITHM_NAMES = ("feddog", "promptonly")


def configure_dataset_args(args: argparse.Namespace) -> None:
    """Set dataset defaults explicitly without a registry layer."""
    if args.dataset == "DomainNet":
        default_root = DOMAINNET_DEFAULT_ROOT
        default_domains = DOMAINNET_DOMAINS
        default_categories = DOMAINNET_CATEGORIES
        image_size = DOMAINNET_IMAGE_SIZE
    elif args.dataset == "PACS":
        default_root = PACS_DEFAULT_ROOT
        default_domains = PACS_DOMAINS
        default_categories = PACS_CATEGORIES
        image_size = PACS_IMAGE_SIZE
    elif args.dataset == "OfficeHome":
        default_root = OFFICEHOME_DEFAULT_ROOT
        default_domains = OFFICEHOME_DOMAINS
        default_categories = OFFICEHOME_CATEGORIES
        image_size = OFFICEHOME_IMAGE_SIZE
    else:
        raise ValueError(f"unsupported dataset: {args.dataset}")

    root = Path(default_root)
    args.data_dir = str(root if root.is_absolute() else ROOT / root)
    args.domains = ",".join(default_domains)
    args.categories = list(default_categories)
    args.num_classes = len(args.categories)
    args.image_size = image_size


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Run FedDoG or the PromptOnly baseline.",
    )

    parser.add_argument("--algorithm", choices=ALGORITHM_NAMES, default="feddog")
    parser.add_argument("--stage", choices=("extract", "generate", "train", "all"), default="all")

    # Dataset/client configuration.
    parser.add_argument("--dataset", choices=DATASET_NAMES, default="DomainNet")
    parser.add_argument("--shots-per-class", type=int, default=16)
    parser.add_argument(
        "--clients-per-domain",
        type=int,
        default=1,
        help="number of clients receiving disjoint balanced shards from each domain",
    )

    parser.add_argument("--local-batch-size", type=int, default=32)
    parser.add_argument("--local-num-workers", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=20)

    # Shared server image generation configuration.
    parser.add_argument("--generated-data-dir", default=None, help="defaults to generated_data/<dataset>")
    parser.add_argument(
        "--images-per-class",
        type=int,
        default=32,
        help="images generated per class/condition; does not limit images read by --stage train",
    )
    parser.add_argument("--generation-batch-size", type=int, default=4)
    parser.add_argument("--prompt-template", default="An image of {}")
    parser.add_argument("--negative-prompt", default=None)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--sd-model", default=None)

    # FedDoG representation and generation configuration.
    parser.add_argument("--granularity", choices=("domain", "class"), default="domain")
    parser.add_argument("--representations-path", default=None)
    parser.add_argument("--clip-model", default="laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
    parser.add_argument("--sd-version", choices=("sd15", "sdxl"), default="sd15")
    parser.add_argument("--image-encoder", default="h94/IP-Adapter")
    parser.add_argument("--image-encoder-subfolder", default=None)
    parser.add_argument("--ip-adapter", default="h94/IP-Adapter")
    parser.add_argument("--ip-adapter-subfolder", default=None)
    parser.add_argument("--ip-adapter-weight", default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument(
        "--disable-daac", action="store_true",
        help="replace domain-aware adaptive conditioning with a fixed IP-Adapter scale",
    )
    parser.add_argument(
        "--fixed-ip-adapter-scale", type=float, default=1.0,
        help="IP-Adapter scale shared by all conditions when --disable-daac is set (0 to 1)",
    )
    parser.add_argument("--entropy-threshold", type=float, default=0.3)
    parser.add_argument("--normalize-entropy", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--filter-top-k",
        type=int,
        choices=(1, 5),
        default=1,
        help="accept generated images when the target class is within CLIP's top-k predictions",
    )
    parser.add_argument("--disable-filter", action="store_true")
    parser.add_argument("--max-attempts", type=int, default=50)
    parser.add_argument("--log-rejections", action="store_true")

    # Global classifier/server configuration.
    parser.add_argument("--synthetic-data-dir", default=None, help="generated run directory; inferred after generation")
    parser.add_argument("--backbone", choices=("resnet18", "resnet50"), default="resnet18")
    parser.add_argument("--pretrained-backbone", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--server-classifier-epochs", type=int, default=50)
    parser.add_argument("--server-classifier-batch-size", type=int, default=64)
    parser.add_argument("--server-classifier-test-batch-size", type=int, default=128)
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=None,
        help="randomly sample this many synthetic images per class for global classifier training; default uses all",
    )
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--checkpoint-path", default=None, help="global model path used with --save-global-classifier")
    parser.add_argument(
        "--results-path",
        default=None,
        help="defaults to outputs/results/<dataset>/<generation-run-and-training-parameters>.json",
    )
    parser.add_argument(
        "--save-global-classifier",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="save the trained global classifier; metrics are always saved",
    )

    # Runtime configuration shared by every stage.
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--cpu-offload", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--disable-diffusion-progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    configure_dataset_args(args)

    if args.algorithm == "feddog":
        algorithm = FedDoG()
    elif args.algorithm == "promptonly":
        algorithm = PromptOnly()
    else:
        raise ValueError(f"unsupported algorithm: {args.algorithm}")

    result = algorithm.run(args)
    if result:
        print("Completed artifacts:")
        for name, value in result.items():
            print(f"  {name}: {value}")


if __name__ == "__main__":
    main()
