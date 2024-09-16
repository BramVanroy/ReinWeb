"""
This file contains the code used to process and create the
ReinWeb dataset. Heavily inspired by https://raw.githubusercontent.com/huggingface/datatrove/main/examples/fineweb.py
"""

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

from rein.utils import STOP_WORDS_DUTCH


"""
GOPHER and default values (English) -> Dutch values

max_avg_word_length: int | None = 10,   --> all: 16 (99 percentile)
max_non_alpha_words_ratio: float | None = 0.8,  --> all: 0.806 (mean)


FEATURES FINEWEB and default values (English) -> Dutch values

line_punct_thr: float = 0.12,  -> 0.088 (wiki: 5 percentile)
line_punct_exclude_zero: bool = False,
short_line_thr: float = 0.67,
short_line_length: int = 30,  -> 27 (wiki: 5 percentile (95% of cases are longer))
char_duplicates_ratio: float = 0.01, -> 0.034 (wiki: 99 percentile)
new_line_ratio: float = 0.3, -> 0.264 (wiki: 99 percentile)
"""


class BaseConfig(BaseModel):
    tasks: int
    time: str
    mem_per_cpu_gb: int | float = 2
    cpus_per_task: int = 1
    randomize_start_duration: int = 0


class ExtractorCfg(BaseConfig):
    gopher_quality_filter: Optional[dict] = field(default_factory=dict)
    c4_quality_filter: Optional[dict] = field(default_factory=dict)
    fineweb_quality_filter: Optional[dict] = field(default_factory=dict)


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


def main(dump: str, output_path: str, partition: str, pipelines_config: str, venv_path: str | None = None):
    configs = yaml.safe_load(Path(pipelines_config).read_text(encoding="utf-8"))

    extract_cfg = ExtractorCfg(**configs["extractor"])
    mh_cfg = MinhashCfg(**configs["minhash"])
    mh1_cfg = BaseConfig(**configs["mh1"])
    mh2_cfg = BaseConfig(**configs["mh2"])
    mh3_cfg = BaseConfig(**configs["mh3"])
    mh4_cfg = BaseConfig(**configs["mh4"])

    pd_base_filter = f"{output_path}/base_processing"

    main_processing_executor = SlurmPipelineExecutor(
        job_name=f"cc_{dump}",
        pipeline=[
            WarcReader(
                f"s3://commoncrawl/crawl-data/{dump}/segments/",
                glob_pattern="*/warc/*",  # we want the warc files
                default_metadata={"dump": dump},
            ),
            URLFilter(exclusion_writer=JsonlWriter(f"{pd_base_filter}/removed/1_url/{dump}")),
            Trafilatura(favour_precision=True),
            LanguageFilter(languages="nld", language_threshold=0.8),
            GopherRepetitionFilter(
                exclusion_writer=JsonlWriter(f"{pd_base_filter}/removed/3_gopher_rep/{dump}"), language=Languages.dutch
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
        qos=None,
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
    pd_slurm_logs_minhash = "slurm_logs/minhash"

    # this is the original data that we want to deduplicate
    INPUT_READER = JsonlReader(f"{pd_base_filter}/output/{dump}")  # this is the output from the first part

    # stage 1 computes minhash signatures for each task (each task gets a set of files)
    stage1 = SlurmPipelineExecutor(
        job_name=f"mh1_{dump}",
        pipeline=[
            INPUT_READER,
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
        qos=None,
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
        tasks=minhash_config.num_buckets * 50,  # the code supports parallelizing each bucket. here we run 50
        # workers per bucket
        randomize_start_duration=mh2_cfg.randomize_start_duration,
        logging_dir=f"{pd_logs_minhash}/buckets",
        partition=partition,
        time=mh2_cfg.time,
        mem_per_cpu_gb=mh2_cfg.mem_per_cpu_gb,
        cpus_per_task=mh2_cfg.cpus_per_task,  # you can add run more (smaller) tasks if you do not have a lot of memory
        venv_path=venv_path,
        depends=stage1,
        qos=None,
    )

    stage3 = SlurmPipelineExecutor(
        job_name=f"mh3_{dump}",
        pipeline=[
            MinhashDedupCluster(
                input_folder=f"{pd_base_minhash}/{dump}/buckets",
                output_folder=f"{pd_base_minhash}/{dump}/remove_ids",
                config=minhash_config,
            ),
        ],
        tasks=mh3_cfg.tasks,  # this step runs on a single task
        logging_dir=f"{pd_logs_minhash}/clustering",
        partition=partition,
        time=mh3_cfg.time,  # and can also be quite slow. Usually not this slow though
        mem_per_cpu_gb=mh3_cfg.mem_per_cpu_gb,
        cpus_per_task=mh3_cfg.cpus_per_task,  # if you dedup a full dump, you do need a lot of memory for this one
        venv_path=venv_path,
        depends=stage2,
        qos=None,
    )

    stage4 = SlurmPipelineExecutor(
        job_name=f"mh4_{dump}",
        pipeline=[
            INPUT_READER,
            TokensCounter(
                tokenizer_name_or_path="yhavinga/gpt2-medium-dutch"
            ),  # you can remove this one, it's just a nice way to know how many tokens we have
            # before and after dedup
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
        qos=None,
    )

    # launch dedup pipelines
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
    cli_kwargs = vars(cparser.parse_args())
    main(**cli_kwargs)
