#!/usr/bin/env python3
"""Prepare DomainNet from its official site and PACS/OfficeHome from Hugging Face."""

from __future__ import annotations

import argparse
import io
import json
import shutil
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence
from urllib.request import Request, urlopen

from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOMAINNET_BASE_URL = "https://csr.bu.edu/ftp/visda/2019/multi-source"
DOMAINNET_DOMAINS = ("clipart", "infograph", "painting", "quickdraw", "real", "sketch")
DATASET_NAMES = ("DomainNet", "PACS", "OfficeHome")


@dataclass(frozen=True)
class DownloadSpec:
    name: str
    repository: str
    output_name: str
    domain_aliases: dict[str, str]


SPECS = {
    "PACS": DownloadSpec(
        name="PACS",
        repository="flwrlabs/pacs",
        output_name="PACS",
        domain_aliases={},
    ),
    "OfficeHome": DownloadSpec(
        name="OfficeHome",
        repository="flwrlabs/office-home",
        output_name="OfficeHome",
        domain_aliases={
            "Real World": "Real",
            "Real_World": "Real",
            "RealWorld": "Real",
        },
    ),
}


def dataset_name(value: str) -> str:
    aliases = {name.lower(): name for name in DATASET_NAMES}
    try:
        return aliases[value.lower()]
    except KeyError as error:
        raise argparse.ArgumentTypeError(f"unknown dataset: {value}") from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Prepare DomainNet using official archives and cleaned train/test splits; "
            "expand PACS and OfficeHome from Hugging Face into image folders."
        ),
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        type=dataset_name,
        choices=DATASET_NAMES,
        default=list(DATASET_NAMES),
        help="datasets to prepare (default: all three); names are case-insensitive",
    )
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "datasets")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Hugging Face download/cache directory")
    parser.add_argument("--revision", default="main", help="Hugging Face revision for PACS/OfficeHome")
    parser.add_argument("--token", default=None, help="optional Hugging Face access token")
    parser.add_argument(
        "--overwrite", action="store_true", help="redownload DomainNet files and rewrite existing images"
    )
    return parser


def download_file(url: str, destination: Path, overwrite: bool = False) -> None:
    """Download atomically, resuming a partial transfer when the server supports it."""
    if destination.is_file() and not overwrite:
        print(f"[DomainNet] using existing {destination.name}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    offset = partial.stat().st_size if partial.is_file() and not overwrite else 0
    headers = {"User-Agent": "FedDoG-dataset-downloader"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    print(f"[DomainNet] downloading {destination.name} (resume offset={offset})", flush=True)
    with urlopen(Request(url, headers=headers), timeout=60) as response:
        resumed = offset > 0 and response.status == 206
        if resumed and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
            raise RuntimeError(f"invalid resume response for {url}")
        expected = response.headers.get("Content-Length")
        initial = offset if resumed else 0
        total = initial + int(expected) if expected is not None else None
        received = 0
        with partial.open("ab" if resumed else "wb") as handle, tqdm(
            desc=f"Download {destination.name}",
            total=total,
            initial=initial,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            dynamic_ncols=True,
        ) as progress:
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
                received += len(chunk)
                progress.update(len(chunk))
        if expected is not None and received != int(expected):
            raise RuntimeError(f"incomplete download: {url}; rerun the command to resume")
    partial.replace(destination)


def extract_domainnet_archive(archive: Path, destination: Path, overwrite: bool) -> None:
    """Keep official paths and filenames; never expose partially extracted images."""
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as handle:
        for member in tqdm(
            handle.infolist(), desc=f"Extract {archive.name}", unit="file", dynamic_ncols=True
        ):
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(destination):
                raise ValueError(f"unsafe archive path: {member.filename!r}")
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if target.is_file() and not overwrite:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_name(target.name + ".part")
            with handle.open(member) as source, partial.open("wb") as output:
                shutil.copyfileobj(source, output)
            partial.replace(target)


def validate_domainnet_split(path: Path, destination: Path, domain: str) -> int:
    """Check every official split entry points to an extracted image."""
    count = 0
    lines = path.read_text().splitlines()
    for line_number, line in enumerate(
        tqdm(lines, desc=f"Verify {path.name}", unit="image", dynamic_ncols=True), start=1
    ):
        fields = line.rsplit(maxsplit=1)
        if len(fields) != 2:
            raise ValueError(f"invalid split entry: {path}:{line_number}")
        relative, label = fields
        parts = Path(relative).parts
        if len(parts) != 3 or parts[0] != domain or ".." in parts or not 0 <= int(label) < 345:
            raise ValueError(f"invalid DomainNet path or label: {path}:{line_number}")
        if not (destination / relative).is_file():
            raise FileNotFoundError(f"split image not found: {destination / relative}")
        count += 1
    if not count:
        raise ValueError(f"empty DomainNet split: {path}")
    return count


def prepare_domainnet(args: argparse.Namespace) -> dict[str, Any]:
    destination = args.output_root.expanduser().resolve() / "DomainNet"
    counts: dict[str, int] = {}
    for domain in DOMAINNET_DOMAINS:
        image_base = f"{DOMAINNET_BASE_URL}/groundtruth" if domain in {"clipart", "painting"} else DOMAINNET_BASE_URL
        archive = destination / "archives" / f"{domain}.zip"
        download_file(f"{image_base}/{domain}.zip", archive, args.overwrite)
        print(f"[DomainNet] extracting {archive.name}", flush=True)
        extract_domainnet_archive(archive, destination, args.overwrite)
        for split in ("train", "test"):
            filename = f"{domain}_{split}.txt"
            path = destination / "splits" / filename
            download_file(f"{DOMAINNET_BASE_URL}/domainnet/txt/{filename}", path, args.overwrite)
            counts[f"{domain}/{split}"] = validate_domainnet_split(path, destination, domain)
        print(f"[DomainNet] verified {domain}: train={counts[f'{domain}/train']} test={counts[f'{domain}/test']}")
    manifest = {
        "dataset": "DomainNet",
        "source": "https://ai.bu.edu/DomainNet/",
        "version": "cleaned",
        "destination": str(destination),
        "split_counts": counts,
    }
    with (destination / "download_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    print(f"[DomainNet] complete: destination={destination}")
    return manifest


def load_huggingface_dataset(
    spec: DownloadSpec, args: argparse.Namespace
) -> tuple[Iterable[dict[str, Any]], list[str]]:
    try:
        from datasets import Image as HFImage
        from datasets import load_dataset
    except (ImportError, AttributeError) as error:
        raise RuntimeError(
            "The Hugging Face 'datasets' package is required. Install project dependencies "
            "or run: pip install datasets"
        ) from error

    dataset = load_dataset(
        spec.repository,
        split="train",
        cache_dir=None if args.cache_dir is None else str(args.cache_dir.expanduser()),
        revision=args.revision,
        token=args.token,
    )
    label_feature = dataset.features["label"]
    label_names = list(getattr(label_feature, "names", []))
    if not label_names:
        raise RuntimeError(f"{spec.repository} does not expose ClassLabel names")
    # Decode bytes ourselves with Pillow. This avoids optional video/image decoder
    # dependencies and makes the extraction behavior stable across datasets versions.
    dataset = dataset.cast_column("image", HFImage(decode=False))
    return dataset, label_names


def decode_image(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            with Image.open(io.BytesIO(value["bytes"])) as image:
                return image.convert("RGB")
        if value.get("path"):
            with Image.open(value["path"]) as image:
                return image.convert("RGB")
    raise TypeError(f"unsupported Hugging Face image value: {type(value)!r}")


def safe_component(value: str, field: str) -> str:
    result = value.strip()
    if not result or result in {".", ".."} or "/" in result or "\\" in result:
        raise ValueError(f"unsafe {field}: {value!r}")
    return result


def materialize_dataset(
    spec: DownloadSpec,
    records: Iterable[dict[str, Any]],
    label_names: Sequence[str],
    output_root: Path,
    overwrite: bool,
) -> dict[str, Any]:
    destination = output_root.expanduser().resolve() / spec.output_name
    destination.mkdir(parents=True, exist_ok=True)
    counters: defaultdict[tuple[str, str], int] = defaultdict(int)
    written = 0
    existing = 0

    for record in tqdm(records, desc=f"Export {spec.name}", unit="image", dynamic_ncols=True):
        domain_value = str(record["domain"])
        domain = safe_component(spec.domain_aliases.get(domain_value, domain_value), "domain")
        raw_label = record["label"]
        class_name = str(raw_label) if isinstance(raw_label, str) else label_names[int(raw_label)]
        class_name = safe_component(class_name, "category")
        key = (domain, class_name)
        image_index = counters[key]
        counters[key] += 1
        image_path = destination / domain / class_name / f"{image_index:06d}.jpg"
        if image_path.is_file() and not overwrite:
            existing += 1
            continue
        image_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = image_path.with_suffix(".jpg.part")
        image = decode_image(record["image"])
        image.save(temporary_path, format="JPEG", quality=95)
        temporary_path.replace(image_path)
        written += 1

    counts = {f"{domain}/{category}": count for (domain, category), count in sorted(counters.items())}
    manifest = {
        "dataset": spec.name,
        "repository": spec.repository,
        "destination": str(destination),
        "total_images": sum(counts.values()),
        "written": written,
        "existing": existing,
        "counts": counts,
    }
    manifest_path = destination / "download_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    print(
        f"[{spec.name}] complete: total={manifest['total_images']} written={written} "
        f"existing={existing} destination={destination}"
    )
    return manifest


def main() -> None:
    args = build_parser().parse_args()
    for name in dict.fromkeys(args.datasets):
        if name == "DomainNet":
            prepare_domainnet(args)
            continue
        spec = SPECS[name]
        print(f"[{name}] downloading {spec.repository} from Hugging Face")
        records, label_names = load_huggingface_dataset(spec, args)
        materialize_dataset(spec, records, label_names, args.output_root, args.overwrite)


if __name__ == "__main__":
    main()

