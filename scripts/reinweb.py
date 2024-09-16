"""
This file contains the code used to process and create the
ReinWeb dataset. Heavily inspired by https://raw.githubusercontent.com/huggingface/datatrove/main/examples/fineweb.py
"""

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


def main(
    dump: str,
    output_path: str,
    partition: str,
    download_extract_tasks: int = 8000,
    minhash_tasks: int = 1000,
    time_extractor: str = "10:00:00",
    time_mh1: str = "5:00:00",
    time_mh2: str = "02:00:00",
    time_mh3: str = "30:00:00",
    time_mh4: str = "5:00:00",
):
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
            LanguageFilter(languages="nld"),
            GopherRepetitionFilter(
                exclusion_writer=JsonlWriter(f"{pd_base_filter}/removed/3_gopher_rep/{dump}"), language=Languages.dutch
            ),
            GopherQualityFilter(
                exclusion_writer=JsonlWriter(f"{pd_base_filter}/removed/4_gopher_qual/{dump}"),
                max_avg_word_length=16,
                max_non_alpha_words_ratio=0.806,
                language=Languages.dutch,
                stop_words=STOP_WORDS_DUTCH,
            ),
            C4QualityFilter(
                filter_no_terminal_punct=False,
                exclusion_writer=JsonlWriter(f"{pd_base_filter}/removed/5_c4/{dump}"),
                language=Languages.dutch,
            ),
            FineWebQualityFilter(
                exclusion_writer=JsonlWriter(f"{pd_base_filter}/removed/6_fineweb_qual/{dump}"),
                language=Languages.dutch,
                line_punct_thr=0.088,
                short_line_thr=0.67,
                short_line_length=27,
                char_duplicates_ratio=0.034,
                new_line_ratio=0.264,
            ),
            JsonlWriter(f"{pd_base_filter}/output/{dump}"),
        ],
        tasks=download_extract_tasks,
        time=time_extractor,
        logging_dir=f"{output_path}/logs/base_processing/{dump}",
        slurm_logs_folder=f"reinweb-logs/base_processing/{dump}/slurm_logs",  # must be local
        randomize_start_duration=180,  # don't hit the bucket all at once with the list requests
        mem_per_cpu_gb=2,
        partition=partition,
    )
    main_processing_executor.run()

    """
        we then applied minhash deduplication to each individual dump,
    """

    # you can also change ngrams or the number of buckets and their size here
    minhash_config = MinhashConfig(
        hash_config=HashConfig(
            hash_fc="sha1",  # better precision -> fewer false positives (collisions)
            precision=64,
        ),
        num_buckets=14,
        hashes_per_bucket=8,
        n_grams=5,
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
        tasks=minhash_tasks,
        time=time_mh1,
        partition=partition,
        logging_dir=f"{pd_logs_minhash}/signatures",
        slurm_logs_folder=f"{pd_slurm_logs_minhash}/signatures/slurm_logs",
        randomize_start_duration=180,
        depends=main_processing_executor,  # only start after the first one completes
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
        randomize_start_duration=180,
        logging_dir=f"{pd_logs_minhash}/buckets",
        partition=partition,
        time=time_mh2,
        mem_per_cpu_gb=4,
        cpus_per_task=3,  # you can add run more (smaller) tasks if you do not have a lot of memory
        depends=stage1,
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
        tasks=1,  # this step runs on a single task
        logging_dir=f"{pd_logs_minhash}/clustering",
        partition=partition,
        time=time_mh3,  # and can also be quite slow. Usually not this slow though
        mem_per_cpu_gb=25,
        cpus_per_task=8,  # if you dedup a full dump, you do need a lot of memory for this one
        depends=stage2,
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
        tasks=minhash_tasks,
        logging_dir=f"{pd_logs_minhash}/filtering",
        partition=partition,
        time=time_mh4,
        mem_per_cpu_gb=4,
        depends=stage3,
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
    cparser.add_argument(
        "--download_extract_tasks", type=int, default=8000, help="Number of download and extract tasks"
    )
    cparser.add_argument("--minhash_tasks", type=int, default=1000, help="Number of minhash tasks")
    cparser.add_argument("--partition", type=str, default="gpu", help="Slurm partition")
    cparser.add_argument("--time_extractor", type=str, default="10:00:00", help="Time for the extractor")
    cparser.add_argument("--time_mh1", type=str, default="5:00:00", help="Time for the first minhash stage")
    cparser.add_argument("--time_mh2", type=str, default="02:00:00", help="Time for the second minhash stage")
    cparser.add_argument("--time_mh3", type=str, default="30:00:00", help="Time for the third minhash stage")
    cparser.add_argument("--time_mh4", type=str, default="5:00:00", help="Time for the fourth minhash stage")
    cli_kwargs = vars(cparser.parse_args())
    main(**cli_kwargs)
