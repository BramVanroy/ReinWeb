import json
from collections import Counter
from os import PathLike
from pathlib import Path

from datasets import Dataset


def main(stats_json_dir: str | PathLike):
    pfiles = list(Path(stats_json_dir).glob("*.json"))
    data = [
        {**{k: v if v else None for k, v in d.items()}, "fname": pfile.stem}
        for pfile in pfiles
        for d in json.loads(pfile.read_text(encoding="utf-8"))
    ]
    ds = Dataset.from_list(data)
    ds = ds.filter(lambda x: x["stats"] is not None, num_proc=4)
    task_names = ds.unique("name")
    df = ds.to_pandas()

    filter_task_d = {task_name: Counter() for task_name in task_names}
    for i, row in df.iterrows():
        task_name = row["name"]
        stats = row["stats"]
        if not stats:
            continue

        if task_name == "📖 - READER: 🕷 Warc":
            filter_task_d[task_name]["forwarded"] += stats["documents"]["total"] or 0
        elif "FORMAT" in task_name:
            filter_task_d[task_name]["received"] += stats["total"] or 0
            filter_task_d[task_name]["forwarded"] += stats["total"] or 0
        else:
            try:
                filter_task_d[task_name]["received"] += stats["total"] or 0
            except KeyError:
                pass
            try:
                filter_task_d[task_name]["dropped"] += stats["dropped"] or 0
            except KeyError:
                pass
            try:
                filter_task_d[task_name]["forwarded"] += stats["forwarded"] or 0
            except KeyError:
                pass

    print(json.dumps(filter_task_d, indent=4))


if __name__ == "__main__":
    import argparse

    cparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    cparser.add_argument(
        "stats_json_dir",
        help="Datatrove logging directory that contains the `stats` (typically a path ending with `/stats`)",
    )

    cli_kwargs = vars(cparser.parse_args())

    main(**cli_kwargs)
