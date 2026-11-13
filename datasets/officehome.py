from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import DataLoader

from .utils import build_record_loader, read_folder_client, read_folder_test, validate_domains

OFFICEHOME_DOMAINS = ("Art", "Clipart", "Product", "Real")
# FedBiP's active 20-class OfficeHome subset, sorted exactly as its config.py does.
OFFICEHOME_CATEGORIES = (
    "Bucket",
    "Chair",
    "Couch",
    "Fan",
    "Fork",
    "Hammer",
    "Kettle",
    "Knives",
    "Marker",
    "Monitor",
    "Mop",
    "Notebook",
    "Oven",
    "Pen",
    "Pencil",
    "Sink",
    "Speaker",
    "Spoon",
    "Telephone",
    "Toys",
)
OFFICEHOME_DEFAULT_ROOT = "datasets/OfficeHome"
OFFICEHOME_IMAGE_SIZE = 256
OFFICEHOME_TEST_SAMPLES_PER_CLASS = 8


def parse_domains(value: str) -> list[str]:
    return validate_domains(value, OFFICEHOME_DOMAINS, "OfficeHome")


def read_client_split(
    root: Path,
    domain: str,
    class_names: Sequence[str],
    shots_per_class: int,
    client_index: int = 0,
    clients_per_domain: int = 1,
) -> tuple[list[tuple[Path, int]], dict[int, str]]:
    return read_folder_client(
        root, domain, class_names, shots_per_class,
        client_index=client_index,
        clients_per_domain=clients_per_domain,
        test_samples_per_class=OFFICEHOME_TEST_SAMPLES_PER_CLASS,
    )


def read_test_split(root: Path, domain: str, class_names: Sequence[str]) -> list[tuple[Path, int]]:
    return read_folder_test(root, domain, class_names, OFFICEHOME_TEST_SAMPLES_PER_CLASS)


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
    """Build OfficeHome loaders for centralized server classifier training."""
    train_loader = build_record_loader(
        train_records,
        OFFICEHOME_IMAGE_SIZE,
        batch_size,
        num_workers,
        device,
        shuffle=True,
        augment=augment,
        drop_last=drop_last,
    )
    test_loaders = {
        domain: build_record_loader(
            read_test_split(root, domain, class_names),
            OFFICEHOME_IMAGE_SIZE,
            test_batch_size,
            num_workers,
            device,
            shuffle=False,
            augment=False,
        )
        for domain in domains
    }
    return train_loader, test_loaders
