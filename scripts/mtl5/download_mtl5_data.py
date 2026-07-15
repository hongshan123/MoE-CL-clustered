#!/usr/bin/env python3
"""Download public MTL5 source datasets and convert them to MoE-CL JSON files."""

import argparse
import json
from pathlib import Path
from typing import Callable, Dict, Iterable

from datasets import load_dataset


DEFAULT_OUTPUT = Path("/home/star/disk-7t/niuxiangqi/data/mtl15")

LABELS: Dict[str, list[str]] = {
    "agnews": ["World", "Sports", "Business", "Science or Technology"],
    "amazon": ["very negative", "negative", "neutral", "positive", "very positive"],
    "dbpedia": [
        "Company", "Educational Institution", "Artist", "Athlete", "Office Holder",
        "Mean of Transportation", "Building", "Natural Place", "Village", "Animal",
        "Plant", "Album", "Film", "Written Work",
    ],
    "yahoo": [
        "Society & Culture", "Science & Mathematics", "Health", "Education & Reference",
        "Computers & Internet", "Sports", "Business & Finance", "Entertainment & Music",
        "Family & Relationships", "Politics & Government",
    ],
}


def format_dbpedia(row: dict) -> str:
    return f"{row['title']}\n{row['content']}"


DATASETS: Dict[str, tuple[str, Callable[[dict], str]]] = {
    "agnews": ("fancyzhx/ag_news", lambda row: row["text"]),
    "amazon": ("SetFit/amazon_reviews_multi_en", lambda row: row["text"]),
    "dbpedia": ("fancyzhx/dbpedia_14", format_dbpedia),
    "yahoo": ("mteb/yahoo_answers_topics", lambda row: row["text"]),
}


def write_json(path: Path, rows: Iterable[dict], dataset_name: str, text_fn: Callable[[dict], str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = LABELS[dataset_name]
    count = 0
    with path.open("w", encoding="utf-8") as file:
        file.write("[")
        for row in rows:
            if count:
                file.write(",")
            json.dump(
                {"sentence": text_fn(row), "label": labels[int(row["label"])]},
                file,
                ensure_ascii=False,
            )
            count += 1
        file.write("]")
    print(f"Wrote {path} ({count} rows)")


def download_dataset(dataset_name: str, output_dir: Path) -> None:
    repo_id, text_fn = DATASETS[dataset_name]
    dataset = load_dataset(repo_id)
    test_split = dataset["test"].train_test_split(test_size=0.5, seed=42, shuffle=True)
    destination = output_dir / dataset_name

    write_json(destination / "train.json", dataset["train"], dataset_name, text_fn)
    write_json(destination / "dev.json", test_split["train"], dataset_name, text_fn)
    write_json(destination / "test.json", test_split["test"], dataset_name, text_fn)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--datasets", nargs="+", choices=sorted(DATASETS), default=sorted(DATASETS)
    )
    args = parser.parse_args()

    for dataset_name in args.datasets:
        print(f"Downloading {dataset_name}...")
        download_dataset(dataset_name, args.output)


if __name__ == "__main__":
    main()
