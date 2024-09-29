"""
This file contains the code used to process and create the
ReinWeb dataset. Heavily inspired by https://raw.githubusercontent.com/huggingface/datatrove/main/examples/fineweb.py
"""

import logging
import os
from dataclasses import field
from pathlib import Path
from typing import Literal, Optional

import datatrove
import yaml
from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.dedup import MinhashDedupCluster, MinhashDedupFilter, MinhashDedupSignature
from datatrove.pipeline.dedup.minhash import MinhashConfig, MinhashDedupBuckets
from datatrove.pipeline.extractors import Trafilatura
from datatrove.pipeline.filters import (
    C4QualityFilter,
    FineWebQualityFilter,
    GopherQualityFilter,
    GopherRepetitionFilter,
    LanguageFilter,
    URLFilter,
)
from datatrove.pipeline.formatters import PIIFormatter
from datatrove.pipeline.readers import JsonlReader, WarcReader
from datatrove.pipeline.tokens import TokensCounter
from datatrove.pipeline.writers.jsonl import JsonlWriter
from datatrove.utils.hashing import HashConfig
from datatrove.utils.typeshelper import Languages
from pydantic import BaseModel

from rein.reinweb_lines_formatter import ReinwebLinesFilter
from rein.reinweb_quality_filter import ReinWebEmptyDocFilter, ReinWebQualityFilter
from rein.utils import STOP_WORDS_DUTCH


class BaseConfig(BaseModel):
    tasks: int
    time: str
    mem_per_cpu_gb: int = 2
    cpus_per_task: int = 1
    randomize_start_duration: int = 0
    do_pipeline: bool = True


class ExtractorCfg(BaseConfig):
    lang_filter_language_threshold: float = 0.65
    gopher_quality_filter: Optional[dict] = field(default_factory=dict)
    c4_quality_filter: Optional[dict] = field(default_factory=dict)
    fineweb_quality_filter: Optional[dict] = field(default_factory=dict)
    reinweb_quality_filter: Optional[dict] = field(default_factory=dict)


class HashCfg(BaseModel):
    precision: Literal[32, 64] = 64
    hash_fc: Literal["sha1", "xxhash"] = "sha1"


class MinhashCfg(BaseModel):
    hash_config: HashCfg
    num_buckets: int
    hashes_per_bucket: int
    n_grams: int


# Add Dutch policy substrings to the C4 filters
datatrove.pipeline.filters.c4_filters.POLICY_SUBSTRINGS = set(
    datatrove.pipeline.filters.c4_filters.POLICY_SUBSTRINGS
    + [
        "gebruik cookies",
        "cookies aanvaarden",
        "gebruik van cookies",
        "cookies weigeren",
        "gebruiksvoorwaarden",
        "privacybeleid",
        "cookiebeleid",
    ]
)


def print_system_stats():
    """
    Print out the number of CPU cores on the system as well as the available memory.
    """
    print(f"Number of CPU cores: {os.cpu_count()}")
    print(f"Available memory: {os.popen('free -h').read()}")


def main(
    dump: str,
    output_path: str,
    partition: str,
    pipelines_config: str,
    venv_path: str | None = None,
    account: str | None = None,
):
    print_system_stats()
    configs = yaml.safe_load(Path(pipelines_config).read_text(encoding="utf-8"))
    sbatch_args = {"account": account} if account else {}

    extract_cfg = ExtractorCfg(**configs["extractor"])
    mh_cfg = MinhashCfg(**configs["minhash"])
    mh1_cfg = BaseConfig(**configs["mh1"])
    mh2_cfg = BaseConfig(**configs["mh2"])
    mh3_cfg = BaseConfig(**configs["mh3"])
    mh4_cfg = BaseConfig(**configs["mh4"])

    pd_base_filter = f"{output_path}/base_processing"

    main_processing_executor = None
    if extract_cfg.do_pipeline:
        main_processing_executor = SlurmPipelineExecutor(
            job_name=f"cc_{dump}",
            pipeline=[
                WarcReader(
                    f"s3://commoncrawl/crawl-data/{dump}/segments/",
                    glob_pattern="*/warc/*",
                    default_metadata={"dump": dump},
                ),
                URLFilter(exclusion_writer=JsonlWriter(f"{pd_base_filter}/removed/1_url/{dump}")),
                Trafilatura(favour_precision=True, timeout=600.0),
                ReinwebLinesFilter(),
                # Empty docs are possible after ReinwebLinesFilter
                ReinWebEmptyDocFilter(exclusion_writer=JsonlWriter(f"{pd_base_filter}/removed/2_empty_doc/{dump}")),
                LanguageFilter(languages=["nl"], language_threshold=extract_cfg.lang_filter_language_threshold),
                GopherRepetitionFilter(
                    exclusion_writer=JsonlWriter(f"{pd_base_filter}/removed/3_gopher_rep/{dump}"),
                    language=Languages.dutch,
                ),
                GopherQualityFilter(
                    exclusion_writer=JsonlWriter(f"{pd_base_filter}/removed/4_gopher_qual/{dump}"),
                    language=Languages.dutch,
                    stop_words=STOP_WORDS_DUTCH,
                    **extract_cfg.gopher_quality_filter,
                ),
                C4QualityFilter(
                    filter_no_terminal_punct=False,
                    exclusion_writer=JsonlWriter(f"{pd_base_filter}/removed/5_c4/{dump}"),
                    language=Languages.dutch,
                    **extract_cfg.c4_quality_filter,
                ),
                FineWebQualityFilter(
                    exclusion_writer=JsonlWriter(f"{pd_base_filter}/removed/6_fineweb_qual/{dump}"),
                    language=Languages.dutch,
                    **extract_cfg.fineweb_quality_filter,
                ),
                ReinWebQualityFilter(
                    exclusion_writer=JsonlWriter(f"{pd_base_filter}/removed/7_reinweb_qual/{dump}"),
                    language=Languages.dutch,
                    **extract_cfg.reinweb_quality_filter,
                ),
                JsonlWriter(f"{pd_base_filter}/output/{dump}"),
            ],
            tasks=extract_cfg.tasks,
            time=extract_cfg.time,
            logging_dir=f"{output_path}/logs/base_processing/{dump}",
            slurm_logs_folder=f"reinweb-logs/base_processing/{dump}/slurm_logs",  # must be local
            randomize_start_duration=extract_cfg.randomize_start_duration,  # don't hit the bucket all at once with the list requests
            mem_per_cpu_gb=extract_cfg.mem_per_cpu_gb,
            partition=partition,
            venv_path=venv_path,
            qos="",
            sbatch_args=sbatch_args,
        )
        main_processing_executor.run()

    """
        we then applied minhash deduplication to each individual dump,
    """

    # you can also change ngrams or the number of buckets and their size here
    minhash_config = MinhashConfig(
        hash_config=HashConfig(
            hash_fc=mh_cfg.hash_config.hash_fc,  # better precision -> fewer false positives (collisions)
            precision=mh_cfg.hash_config.precision,
        ),
        num_buckets=mh_cfg.num_buckets,
        hashes_per_bucket=mh_cfg.hashes_per_bucket,
        n_grams=mh_cfg.n_grams,
    )
    pd_base_minhash = f"{output_path}/minhash"
    pd_logs_minhash = f"{output_path}/logs/minhash"
    pd_slurm_logs_minhash = str(Path(__file__).parent.parent / "slurm_logs" / "minhash")

    # this is the original data that we want to deduplicate
    input_reader = JsonlReader(f"{pd_base_filter}/output/{dump}")  # this is the output from the first part

    # stage 1 computes minhash signatures for each task (each task gets a set of files)
    stage1 = None
    if mh1_cfg.do_pipeline:
        stage1 = SlurmPipelineExecutor(
            job_name=f"mh1_{dump}",
            pipeline=[
                input_reader,
                MinhashDedupSignature(output_folder=f"{pd_base_minhash}/{dump}/signatures", config=minhash_config),
            ],
            tasks=mh1_cfg.tasks,
            time=mh1_cfg.time,
            partition=partition,
            logging_dir=f"{pd_logs_minhash}/signatures",
            slurm_logs_folder=f"{pd_slurm_logs_minhash}/signatures/slurm_logs",
            randomize_start_duration=mh1_cfg.randomize_start_duration,
            venv_path=venv_path,
            depends=main_processing_executor,  # only start after the first one completes
            qos="",
            sbatch_args=sbatch_args,
        )
        stage1.run()

    stage2 = None
    if mh2_cfg.do_pipeline:
        if mh2_cfg.tasks != 1:
            logging.warning(
                f"You set the number of tasks for Minhash stage 2 to {mh2_cfg.tasks} but this will not have an effect."
                f" The value is automatically set to 50 * minhash.num_buckets."
                f" If you want to run more tasks, you can increase the number of buckets."
            )

        stage2 = SlurmPipelineExecutor(
            job_name=f"mh2_{dump}",
            pipeline=[
                MinhashDedupBuckets(
                    input_folder=f"{pd_base_minhash}/{dump}/signatures",
                    output_folder=f"{pd_base_minhash}/{dump}/buckets",
                    config=MinhashConfig(hash_config=minhash_config.hash_config),
                ),
            ],
            # the code supports parallelizing each bucket. here we run 50 workers per bucket
            tasks=minhash_config.num_buckets * 50,
            randomize_start_duration=mh2_cfg.randomize_start_duration,
            logging_dir=f"{pd_logs_minhash}/buckets",
            partition=partition,
            time=mh2_cfg.time,
            mem_per_cpu_gb=mh2_cfg.mem_per_cpu_gb,
            # you can add run more (smaller) tasks if you do not have a lot of memory
            cpus_per_task=mh2_cfg.cpus_per_task,
            venv_path=venv_path,
            depends=stage1,
            qos="",
            sbatch_args=sbatch_args,
        )
        stage2.run()

    stage3 = None
    if mh3_cfg.do_pipeline:
        stage3 = SlurmPipelineExecutor(
            job_name=f"mh3_{dump}",
            pipeline=[
                MinhashDedupCluster(
                    input_folder=f"{pd_base_minhash}/{dump}/buckets",
                    output_folder=f"{pd_base_minhash}/{dump}/remove_ids",
                    config=minhash_config,
                ),
            ],
            tasks=mh3_cfg.tasks,
            logging_dir=f"{pd_logs_minhash}/clustering",
            partition=partition,
            time=mh3_cfg.time,
            mem_per_cpu_gb=mh3_cfg.mem_per_cpu_gb,
            # if you dedup a full dump, you do need a lot of memory for this one
            cpus_per_task=mh3_cfg.cpus_per_task,
            venv_path=venv_path,
            depends=stage2,
            qos="",
            sbatch_args=sbatch_args,
        )
        stage3.run()

    if mh4_cfg.do_pipeline:
        stage4 = SlurmPipelineExecutor(
            job_name=f"mh4_{dump}",
            pipeline=[
                input_reader,
                # you can remove this one, it's just a nice way to know how many tokens we have before and after dedup
                TokensCounter(tokenizer_name_or_path="yhavinga/gpt2-medium-dutch"),
                MinhashDedupFilter(input_folder=f"{pd_base_minhash}/{dump}/remove_ids"),
                # run the PII removal
                PIIFormatter(),
                JsonlWriter(f"{pd_base_minhash}/{dump}/deduped_output"),
            ],
            tasks=mh1_cfg.tasks,
            logging_dir=f"{pd_logs_minhash}/filtering",
            partition=partition,
            time=mh4_cfg.time,
            mem_per_cpu_gb=mh4_cfg.mem_per_cpu_gb,
            venv_path=venv_path,
            depends=stage3,
            qos="",
            sbatch_args=sbatch_args,
        )
        stage4.run()


if __name__ == "__main__":
    import argparse

    cparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    cparser.add_argument(
        "--dump", type=str, required=True, help="CommonCrawl dump (see https://commoncrawl.org/overview)"
    )
    cparser.add_argument("--output_path", type=str, required=True, help="Output path")
    cparser.add_argument("--partition", type=str, required=True, help="Slurm partition")
    cparser.add_argument("--pipelines_config", type=str, required=True, help="Path to the pipelines YAML config file")
    cparser.add_argument("--venv_path", type=str, help="Path to the virtual environment")
    cparser.add_argument("--account", type=str, help="Slurm account")

    cli_kwargs = vars(cparser.parse_args())
    main(**cli_kwargs)
