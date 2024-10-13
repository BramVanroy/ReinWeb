from huggingface_hub import create_branch, list_repo_refs, upload_large_folder


def main(local_path: str, hf_repo: str, revision: str | None = None, public: bool = False):
    print(f"Uploading folder {local_path} to {hf_repo} (revision: {revision}; public: {public})")

    if revision:
        refs = list_repo_refs(hf_repo, repo_type="dataset")
        rev_names = [b.name for b in refs.branches]
        if revision not in rev_names:
            print(f"Creating branch {revision}")
            create_branch(hf_repo, repo_type="dataset", branch=revision)

    upload_large_folder(
        repo_id=hf_repo,
        folder_path=local_path,
        revision=revision,
        private=not public,
        repo_type="dataset",
        allow_patterns=["*.jsonl.*", "*.jsonl"],
    )


if __name__ == "__main__":
    import argparse

    cparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    cparser.add_argument("--local_path", type=str, required=True, help="Output path")
    cparser.add_argument("--hf_repo", type=str, required=True, help="HF repo name")
    cparser.add_argument("--revision", type=str, help="HF repo revision/branch")
    cparser.add_argument("--public", action="store_true", help="Make the repo public", default=False)

    cli_kwargs = vars(cparser.parse_args())
    main(**cli_kwargs)
