"""Calculate the heuristics features from Gopher and FineWeb but specifically for Dutch text by using SONAR and wiki data."""

import json
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from pathlib import Path
from statistics import mean

import numpy as np
import spacy
from datasets import Dataset, concatenate_datasets, load_dataset
from datatrove.pipeline.filters.gopher_repetition_filter import find_duplicates
from datatrove.utils.text import PUNCTUATION_SET

from rein.utils import STOP_WORDS_DUTCH


tokenizer = spacy.blank("nl")


@dataclass
class DatasetConfig:
    dataset_name: str
    identifier: str
    dataset_config: str | None = None
    dataset_split: str | None = None
    text_column: str | None = None
    from_hub: bool = True


STOP_CHARS = (".", "'", '"', "!", "?")


def process_doc(sample):
    text = sample["text"]
    doc = tokenizer(text)
    lines = doc.text.split("\n")
    lines = [line.strip() for line in lines if line.strip() != ""]

    # Fineweb
    line_punct_ratio = sum(1 for line in lines if line.endswith(STOP_CHARS)) / len(lines)
    short_line_ratio = sum(1 for line in lines if len(line) <= 30) / len(lines)
    char_dup_ratio = find_duplicates(lines)[1] / len(doc.text.replace("\n", ""))
    list_ratio = text.count("\n") / len(doc)

    # Gopher
    num_tokens = len(doc)
    non_symbol_tokens = [w for w in doc if any(ch not in PUNCTUATION_SET for ch in w.text)]
    num_non_symbol_tokens = len(non_symbol_tokens)
    avg_token_len = mean([len(w.text) for w in non_symbol_tokens]) if num_non_symbol_tokens > 0 else 0.0
    non_alpha_ratio = sum([any((c.isalpha() for c in t.text)) for t in doc]) / num_tokens
    num_stop_words = sum([1 for t in doc if t.text.lower() in STOP_WORDS_DUTCH])

    return {
        "num_lines": len(lines),
        "num_tokens": num_tokens,
        "line_punct_ratio": line_punct_ratio,
        "short_line_ratio": short_line_ratio,
        "char_dup_ratio": char_dup_ratio,
        "list_ratio": list_ratio,
        "gopher_avg_token_len": avg_token_len,
        "gopher_num_non_symbol_tokens": num_non_symbol_tokens,
        "gopher_non_alpha_ratio": non_alpha_ratio,
        "gopher_num_stop_words": num_stop_words,
    }


def get_property_stats(values: list):
    return {
        "min": min(values),
        "0.01-percentile": np.percentile(values, 0.01),
        "0.1-percentile": np.percentile(values, 0.1),
        "1-percentile": np.percentile(values, 1),
        "5-percentile": np.percentile(values, 5),
        "95-percentile": np.percentile(values, 95),
        "99-percentile": np.percentile(values, 99),
        "99.9-percentile": np.percentile(values, 99.9),
        "99.99-percentile": np.percentile(values, 99.99),
        "max": max(values),
        "mean": sum(values) / len(values),
        "std": np.std(values),
        "median": np.median(values),
    }


attrs = [
    "num_lines",
    "num_tokens",
    "line_punct_ratio",
    "short_line_ratio",
    "char_dup_ratio",
    "list_ratio",
    "gopher_avg_token_len",
    "gopher_num_non_symbol_tokens",
    "gopher_non_alpha_ratio",
    "gopher_num_stop_words",
]


def get_subset_attr_stats(subset, attr, subds):
    # Get statistics for the given attribute
    try:
        stats = get_property_stats(subds[attr])
    except ValueError as exc:
        raise ValueError(f"Error while calculating stats for {attr} in {subset} ({subds[attr]})") from exc

    attr_name = f"{attr} ({subset})"
    print(f"{attr_name}\n{'=' * len(attr_name)}")
    print(json.dumps(stats, indent=4))

    return (subset, attr, stats)


def main(dataset_dir, *ds_configs: DatasetConfig, num_proc: int | None = None):
    num_proc = num_proc or cpu_count() - 1
    datasets = []
    for ds_config in ds_configs:
        if ds_config.from_hub:
            ds = load_dataset(ds_config.dataset_name, ds_config.dataset_config, split=ds_config.dataset_split)
            ds = ds.select_columns(ds_config.text_column)
            if ds.column_names[0] != "text":
                ds = ds.rename_column(ds_config.text_column, "text")
        else:
            ds = load_dataset(ds_config.dataset_name)["train"]

        ds = ds.add_column("ds_name", [ds_config.identifier] * len(ds))

        datasets.append(ds)

    print(datasets)
    ds: Dataset = concatenate_datasets(datasets)
    ds = ds.map(process_doc, num_proc=num_proc)
    ds.save_to_disk(dataset_dir)

    subsetnames = ["all"] + [ds_config.identifier for ds_config in ds_configs]
    subdss = {
        subset: ds if subset == "all" else ds.filter(lambda x: x["ds_name"] == subset, num_proc=num_proc)
        for subset in subsetnames
    }

    # Check that ds_names exist
    for ds, subset in zip(subdss.values(), subsetnames):
        if len(ds) == 0:
            raise ValueError(f"No samples found for subset {subset}")

    tasks = [(subname, attr, subds) for subname, subds in subdss.items() for attr in attrs]
    with Pool(min(cpu_count(), len(tasks))) as pool:
        results = pool.starmap(get_subset_attr_stats, tasks)

    all_stats = {subset: {} for subset in subsetnames}
    for subset, attr, stats in results:
        all_stats[subset][attr] = stats

    Path(dataset_dir).joinpath("heuristics.json").write_text(json.dumps(all_stats, indent=4), encoding="utf-8")


if __name__ == "__main__":
    wiki = DatasetConfig("wikimedia/wikipedia", "wiki_nl", "20231101.nl", "train", "text", from_hub=True)
    dpc = DatasetConfig("BramVanroy/DPC1.0-dutch", "dpc", dataset_split="train", text_column="text", from_hub=True)
    sonar = DatasetConfig("/home/local/vanroy/ReinWeb/data/sonar_plaintext", "sonar", from_hub=False)
    main("/home/local/vanroy/ReinWeb/data/heuristics-wiki+dpc+sonar", wiki, dpc, sonar)
