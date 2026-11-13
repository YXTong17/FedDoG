from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import DataLoader

from .utils import balanced_partition, build_record_loader

DOMAINNET_DOMAINS = ("clipart", "infograph", "painting", "quickdraw", "real", "sketch")
# First 20 categories in the official DomainNet label order (labels 0–19).
DOMAINNET_CATEGORIES = (
    "aircraft_carrier",
    "airplane",
    "alarm_clock",
    "ambulance",
    "angel",
    "animal_migration",
    "ant",
    "anvil",
    "apple",
    "arm",
    "asparagus",
    "axe",
    "backpack",
    "banana",
    "bandage",
    "barn",
    "baseball",
    "baseball_bat",
    "basket",
    "basketball",
)
DOMAINNET_DEFAULT_ROOT = "datasets/DomainNet"
DOMAINNET_NUM_CLASSES = len(DOMAINNET_CATEGORIES)
DOMAINNET_IMAGE_SIZE = 256


def parse_domains(value: str) -> list[str]:
    """Parse and validate a comma-separated DomainNet client/domain list."""
    domains = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(domains) - set(DOMAINNET_DOMAINS))
    if unknown:
        raise ValueError(f"unknown DomainNet domains: {unknown}")
    if not domains:
        raise ValueError("at least one DomainNet domain is required")
    return domains


def read_domainnet_client_split(
    root: Path,
    domain: str,
    num_classes: int,
    shots_per_class: int,
    categories: Sequence[str] | None = None,
    client_index: int = 0,
    clients_per_domain: int = 1,
) -> tuple[list[tuple[Path, int]], dict[int, str]]:
    """Read one training client and recover the global label-to-name mapping."""
    split_path = root / "splits" / f"{domain}_train.txt"
    if not split_path.is_file():
        raise FileNotFoundError(f"DomainNet split not found: {split_path}")

    records: list[tuple[Path, int]] = []
    class_names: dict[int, str] = {}
    counts: defaultdict[int, int] = defaultdict(int)
    with split_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            fields = raw_line.strip().rsplit(maxsplit=1)
            if len(fields) != 2:
                raise ValueError(f"invalid line {line_number} in {split_path}")
            relative_name, label_text = fields
            path_parts = Path(relative_name).parts
            if len(path_parts) < 2:
                raise ValueError(f"cannot infer class name from {relative_name!r}")
            original_label = int(label_text)
            category = path_parts[1]
            if categories is None:
                label = original_label
                if label < 0 or label >= num_classes:
                    continue
            else:
                if category not in categories:
                    continue
                label = categories.index(category)
            previous = class_names.setdefault(label, category)
            if previous != category:
                raise ValueError(f"label {label} maps to both {previous!r} and {category!r}")
            if shots_per_class > 0 and counts[label] >= shots_per_class * clients_per_domain:
                continue
            image_path = root / relative_name
            if not image_path.is_file():
                raise FileNotFoundError(f"DomainNet image not found: {image_path}")
            records.append((image_path, label))
            counts[label] += 1
    grouped: defaultdict[int, list[tuple[Path, int]]] = defaultdict(list)
    for record in records:
        grouped[record[1]].append(record)
    client_records: list[tuple[Path, int]] = []
    for label in range(num_classes):
        shard_index = (client_index - label) % clients_per_domain
        client_records.extend(balanced_partition(grouped[label], clients_per_domain, shard_index))
    return client_records, class_names


def read_domainnet_test_split(
    root: Path,
    domain: str,
    class_names: Sequence[str],
) -> list[tuple[Path, int]]:
    """Read and remap DomainNet test labels to the selected category order."""
    split_path = root / "splits" / f"{domain}_test.txt"
    if not split_path.is_file():
        raise FileNotFoundError(f"DomainNet split not found: {split_path}")
    label_by_name = {name: label for label, name in enumerate(class_names)}
    records: list[tuple[Path, int]] = []
    with split_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            fields = raw_line.strip().rsplit(maxsplit=1)
            if len(fields) != 2:
                raise ValueError(f"invalid line {line_number} in {split_path}")
            relative_name, _ = fields
            path_parts = Path(relative_name).parts
            if len(path_parts) < 2:
                raise ValueError(f"cannot infer class name from {relative_name!r}")
            category = path_parts[1]
            if category not in label_by_name:
                continue
            image_path = root / relative_name
            if not image_path.is_file():
                raise FileNotFoundError(f"DomainNet image not found: {image_path}")
            records.append((image_path, label_by_name[category]))
    return records


def build_server_loaders(
    root: Path,
    train_records: Sequence[tuple[Path, int]],
    class_names: Sequence[str],
    domains: Sequence[str],
    batch_size: int,
    test_batch_size: int,
    num_workers: int,
    device: torch.device,
    drop_last: bool = False,
    augment: bool = False,
) -> tuple[DataLoader, dict[str, DataLoader]]:
    """Build DomainNet loaders for centralized server classifier training."""
    train_loader = build_record_loader(
        train_records,
        DOMAINNET_IMAGE_SIZE,
        batch_size,
        num_workers,
        device,
        shuffle=True,
        augment=augment,
        drop_last=drop_last,
    )
    test_loaders = {
        domain: build_record_loader(
            read_domainnet_test_split(root, domain, class_names),
            DOMAINNET_IMAGE_SIZE,
            test_batch_size,
            num_workers,
            device,
            shuffle=False,
            augment=False,
        )
        for domain in domains
    }
    return train_loader, test_loaders
