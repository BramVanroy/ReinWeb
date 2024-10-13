from huggingface_hub import upload_large_folder


def main(local_path: str, hf_repo: str, public: bool = False):
    upload_large_folder(repo_id=hf_repo, folder_path=local_path, private=not public, repo_type="dataset")


if __name__ == "__main__":
    import argparse

    cparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    cparser.add_argument("--local_path", type=str, required=True, help="Output path")
    cparser.add_argument("--hf_repo", type=str, required=True, help="HF repo name")
    cparser.add_argument("--revision", type=str, help="HF repo revision/branch")
    cparser.add_argument("--public", action="store_true", help="Make the repo public", default=False)

    cli_kwargs = vars(cparser.parse_args())
    main(**cli_kwargs)
