from __future__ import annotations

import argparse
from pathlib import Path

from datasets import domainnet, officehome, pacs


ImageRecord = tuple[Path, int]


def configured_client_ids(args: argparse.Namespace) -> list[str]:
    clients_per_domain = args.clients_per_domain
    if clients_per_domain <= 0:
        raise ValueError("--clients-per-domain must be positive")
    return [
        f"{domain}_client_{client_index}"
        for domain in args.domains.split(",")
        for client_index in range(clients_per_domain)
    ]


def collect_client_records(
    args: argparse.Namespace,
) -> tuple[dict[str, list[ImageRecord]], dict[str, str]]:
    """Read balanced, disjoint client shards after the dataset test split is fixed."""
    root = Path(args.data_dir).expanduser().resolve()
    class_names = list(args.categories)
    records_by_client: dict[str, list[ImageRecord]] = {}
    client_domains: dict[str, str] = {}
    clients_per_domain = args.clients_per_domain
    configured_client_ids(args)
    for domain in args.domains.split(","):
        for client_index in range(clients_per_domain):
            client_id = f"{domain}_client_{client_index}"
            if args.dataset == "DomainNet":
                records, _ = domainnet.read_domainnet_client_split(
                    root, domain, args.num_classes, args.shots_per_class, class_names,
                    client_index, clients_per_domain,
                )
            elif args.dataset == "PACS":
                records, _ = pacs.read_client_split(
                    root, domain, class_names, args.shots_per_class,
                    client_index, clients_per_domain,
                )
            elif args.dataset == "OfficeHome":
                records, _ = officehome.read_client_split(
                    root, domain, class_names, args.shots_per_class,
                    client_index, clients_per_domain,
                )
            else:
                raise ValueError(f"unsupported dataset: {args.dataset}")
            records_by_client[client_id] = records
            client_domains[client_id] = domain
    return records_by_client, client_domains
