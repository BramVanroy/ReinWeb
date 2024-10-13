import os

from datasets import disable_caching, load_dataset


disable_caching()


def main(local_path: str, hf_repo: str, config_name: str, max_shard_size: str = "1GB", public: bool = False):
    print(
        f"Uploading folder {local_path} to {hf_repo}"
        f" in remote folder {config_name.replace('--', '/')}"
        f" (config: {config_name}; public: {public})"
    )

    num_cpus = max(os.cpu_count() - 1, 1)
    ds = load_dataset("json", data_files=f"{local_path}/*.jsonl.gz", split="train", num_proc=num_cpus)

    ds.push_to_hub(
        repo_id=hf_repo,
        config_name=config_name,
        data_dir=config_name.replace("--", "/"),  # E.g., CC-MAIN-2019-30--base will be saved in CC-MAIN-2019-30/base
        max_shard_size=max_shard_size,
        private=not public,
        commit_message=f"Upload to {hf_repo} with config {config_name} and max_shard_size {max_shard_size}",
    )


if __name__ == "__main__":
    import argparse

    cparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    cparser.add_argument("--local_path", type=str, required=True, help="Output path")
    cparser.add_argument("--hf_repo", type=str, required=True, help="HF repo name")
    cparser.add_argument("--config_name", type=str, required=True, help="HF repo config_name")
    cparser.add_argument("--max_shard_size", type=str, help="HF repo config_name", default="1GB")
    cparser.add_argument("--public", action="store_true", help="Make the repo public", default=False)

    cli_kwargs = vars(cparser.parse_args())
    main(**cli_kwargs)
