"""Assignment 1, Problem 2.7(d)：把四个语料文件编码为 uint16。

运行全部数据集：

    uv run python scripts/problem_2_7_encode_datasets.py --dataset all

脚本只生成四个 ``.bin`` 文件，并把数据统计更新到
``docs/problem_2_7_results.md``；不会生成 JSON 元数据。
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np

from cs336_basics.tokenizer import Tokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPECIAL_TOKEN = "<|endoftext|>"
TOKENIZED_DATA_DIRECTORY = PROJECT_ROOT / "data" / "tokenized"
RESULTS_PATH = PROJECT_ROOT / "docs" / "problem_2_7_results.md"

TINYSTORIES_TOKENIZER_DIRECTORY = (
    PROJECT_ROOT / "outputs" / "problem_2_5_tinystories" / "final"
)
OPENWEBTEXT_TOKENIZER_DIRECTORY = (
    PROJECT_ROOT / "outputs" / "problem_2_5_openwebtext" / "final"
)

ENCODING_SECTION_START = "<!-- PROBLEM_2_7_D_START -->"
ENCODING_SECTION_END = "<!-- PROBLEM_2_7_D_END -->"
TEXT_BATCH_SIZE = 1024 * 1024
DEFAULT_WORKERS = min(8, max(1, (os.cpu_count() or 1) - 2))

# 每个子进程启动时各自加载一份 tokenizer，之后重复用于所有文本批次。
worker_tokenizer: Tokenizer | None = None


@dataclass(frozen=True)
class DatasetConfig:
    """一个数据集的输入、Tokenizer 参数和输出文件名。"""

    input_path: Path
    tokenizer_directory: Path
    vocab_filename: str
    merges_filename: str
    output_filename: str


# 按文件从小到大排列，让短任务先完成并尽早暴露问题。
DATASETS = {
    "tinystories_valid": DatasetConfig(
        input_path=PROJECT_ROOT / "data" / "TinyStoriesV2-GPT4-valid.txt",
        tokenizer_directory=TINYSTORIES_TOKENIZER_DIRECTORY,
        vocab_filename="tinystories_vocab.pkl",
        merges_filename="tinystories_merges.pkl",
        output_filename="problem_2_7_tinystories_valid.uint16.bin",
    ),
    "openwebtext_valid": DatasetConfig(
        input_path=PROJECT_ROOT / "data" / "owt_valid.txt",
        tokenizer_directory=OPENWEBTEXT_TOKENIZER_DIRECTORY,
        vocab_filename="openwebtext_vocab.pkl",
        merges_filename="openwebtext_merges.pkl",
        output_filename="problem_2_7_openwebtext_valid.uint16.bin",
    ),
    "tinystories_train": DatasetConfig(
        input_path=PROJECT_ROOT / "data" / "TinyStoriesV2-GPT4-train.txt",
        tokenizer_directory=TINYSTORIES_TOKENIZER_DIRECTORY,
        vocab_filename="tinystories_vocab.pkl",
        merges_filename="tinystories_merges.pkl",
        output_filename="problem_2_7_tinystories_train.uint16.bin",
    ),
    "openwebtext_train": DatasetConfig(
        input_path=PROJECT_ROOT / "data" / "owt_train.txt",
        tokenizer_directory=OPENWEBTEXT_TOKENIZER_DIRECTORY,
        vocab_filename="openwebtext_vocab.pkl",
        merges_filename="openwebtext_merges.pkl",
        output_filename="problem_2_7_openwebtext_train.uint16.bin",
    ),
}


def load_tokenizer(config: DatasetConfig) -> Tokenizer:
    """加载当前数据集应该使用的 vocab 和 merges。"""

    vocab_path = config.tokenizer_directory / config.vocab_filename
    merges_path = config.tokenizer_directory / config.merges_filename

    if not vocab_path.is_file() or not merges_path.is_file():
        raise FileNotFoundError(
            "找不到 tokenizer 参数，请检查以下文件：\n"
            f"  {vocab_path}\n"
            f"  {merges_path}"
        )

    tokenizer = Tokenizer.from_files(
        vocab_filepath=vocab_path,
        merges_filepath=merges_path,
        special_tokens=[SPECIAL_TOKEN],
    )

    largest_token_id = max(tokenizer.vocab)
    if largest_token_id > np.iinfo(np.uint16).max:
        raise ValueError(
            f"最大 token ID 是 {largest_token_id}，超过 uint16 上限 65535。"
        )

    return tokenizer


def write_token_buffer(output_file: BinaryIO, token_buffer: list[int]) -> int:
    """把一批 Python 整数转换为 uint16 并写入文件。"""

    token_array = np.asarray(token_buffer, dtype=np.uint16)
    return output_file.write(token_array.tobytes())


def initialize_worker(config: DatasetConfig) -> None:
    """在编码子进程中加载一次 tokenizer。"""

    global worker_tokenizer
    worker_tokenizer = load_tokenizer(config)


def read_text_batches(input_file: object) -> Iterator[list[str]]:
    """把文本行组合成约 1MB 的批次，避免一次读取整个语料。"""

    text_batch: list[str] = []
    batch_character_count = 0

    for line in input_file:
        text_batch.append(line)
        batch_character_count += len(line)

        if batch_character_count >= TEXT_BATCH_SIZE:
            yield text_batch
            text_batch = []
            batch_character_count = 0

    if text_batch:
        yield text_batch


def encode_text_batch(text_batch: list[str]) -> bytes:
    """由一个子进程编码一批文本，并直接返回 uint16 字节。"""

    if worker_tokenizer is None:
        raise RuntimeError("编码子进程尚未加载 tokenizer。")

    token_ids: list[int] = []
    for line in text_batch:
        token_ids.extend(worker_tokenizer.encode(line))

    return np.asarray(token_ids, dtype=np.uint16).tobytes()


def encode_in_one_process(
    input_file: object,
    output_file: BinaryIO,
    tokenizer: Tokenizer,
    buffer_size: int,
    start_time: float,
) -> tuple[int, int]:
    """单进程模式，主要用于小数据验证。"""

    token_buffer: list[int] = []
    token_count = 0
    written_bytes = 0
    next_progress_token_count = 10_000_000

    for token_id in tokenizer.encode_iterable(input_file):
        token_buffer.append(token_id)
        token_count += 1

        if len(token_buffer) >= buffer_size:
            written_bytes += write_token_buffer(output_file, token_buffer)
            token_buffer.clear()

        if token_count >= next_progress_token_count:
            elapsed_seconds = time.perf_counter() - start_time
            print(f"  已编码 {token_count:,} tokens，用时 {elapsed_seconds:.1f} 秒")
            next_progress_token_count += 10_000_000

    if token_buffer:
        written_bytes += write_token_buffer(output_file, token_buffer)

    return token_count, written_bytes


def encode_in_parallel(
    input_file: object,
    output_file: BinaryIO,
    config: DatasetConfig,
    workers: int,
    start_time: float,
) -> tuple[int, int]:
    """多进程编码文本批次；imap 保证结果顺序与输入顺序一致。"""

    token_count = 0
    written_bytes = 0
    next_progress_token_count = 10_000_000
    process_context = multiprocessing.get_context("spawn")

    with process_context.Pool(
        processes=workers,
        initializer=initialize_worker,
        initargs=(config,),
    ) as process_pool:
        encoded_batches = process_pool.imap(
            encode_text_batch,
            read_text_batches(input_file),
            chunksize=1,
        )

        for encoded_batch in encoded_batches:
            output_file.write(encoded_batch)
            written_bytes += len(encoded_batch)
            token_count += len(encoded_batch) // np.dtype(np.uint16).itemsize

            while token_count >= next_progress_token_count:
                elapsed_seconds = time.perf_counter() - start_time
                print(f"  已编码 {token_count:,} tokens，用时 {elapsed_seconds:.1f} 秒")
                next_progress_token_count += 10_000_000

    return token_count, written_bytes


def encode_dataset(
    dataset_name: str,
    config: DatasetConfig,
    output_directory: Path = TOKENIZED_DATA_DIRECTORY,
    overwrite: bool = False,
    buffer_size: int = 1_000_000,
    workers: int = 1,
) -> dict[str, float | int | str]:
    """流式编码一个数据集，成功后把临时文件原子替换为最终文件。"""

    if not config.input_path.is_file():
        raise FileNotFoundError(f"找不到数据集：{config.input_path}")
    if buffer_size <= 0:
        raise ValueError("buffer_size 必须大于 0")
    if workers <= 0:
        raise ValueError("workers 必须大于 0")

    tokenizer = load_tokenizer(config)
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / config.output_filename
    partial_path = Path(f"{output_path}.partial")

    existing_paths = [path for path in (output_path, partial_path) if path.exists()]
    if existing_paths and not overwrite:
        formatted_paths = "\n".join(f"  {path}" for path in existing_paths)
        raise FileExistsError(
            "以下编码文件已经存在。确认重做时请增加 --overwrite：\n"
            f"{formatted_paths}"
        )

    start_time = time.perf_counter()

    print(f"开始编码 {dataset_name}：{config.input_path}")
    print(f"编码进程数：{workers}")

    with (
        config.input_path.open("r", encoding="utf-8") as input_file,
        partial_path.open("wb") as output_file,
    ):
        if workers == 1:
            token_count, written_bytes = encode_in_one_process(
                input_file=input_file,
                output_file=output_file,
                tokenizer=tokenizer,
                buffer_size=buffer_size,
                start_time=start_time,
            )
        else:
            token_count, written_bytes = encode_in_parallel(
                input_file=input_file,
                output_file=output_file,
                config=config,
                workers=workers,
                start_time=start_time,
            )

    os.replace(partial_path, output_path)
    elapsed_seconds = time.perf_counter() - start_time
    input_bytes = config.input_path.stat().st_size

    print(f"完成 {dataset_name}：{token_count:,} tokens，{elapsed_seconds:.1f} 秒")
    return {
        "dataset": dataset_name,
        "token_count": token_count,
        "input_bytes": input_bytes,
        "output_bytes": written_bytes,
        "bytes_per_token": input_bytes / token_count,
        "elapsed_seconds": elapsed_seconds,
        "output_path": str(output_path),
    }


def build_encoding_section(output_directory: Path) -> str:
    """根据实际存在的四个 .bin 文件生成(d)结果表格。"""

    rows: list[str] = []

    for dataset_name, config in DATASETS.items():
        output_path = output_directory / config.output_filename

        if not output_path.is_file():
            rows.append(f"| {dataset_name} | 未生成 | — | — | — |")
            continue

        output_bytes = output_path.stat().st_size
        if output_bytes % np.dtype(np.uint16).itemsize != 0:
            raise ValueError(f"{output_path} 大小不是 uint16 的整数倍，文件可能损坏。")

        token_count = output_bytes // np.dtype(np.uint16).itemsize
        input_bytes = config.input_path.stat().st_size
        bytes_per_token = input_bytes / token_count
        rows.append(
            f"| {dataset_name} | 完成 | {input_bytes:,} | "
            f"{token_count:,} | {bytes_per_token:.4f} |"
        )

    rows_text = "\n".join(rows)
    return f"""{ENCODING_SECTION_START}
四个文件均使用对应语料训练得到的 tokenizer，并以原始 `uint16` 顺序保存，可由 `np.memmap` 直接读取。

| 数据集 | 状态 | 原始字节数 | Token 数 | bytes/token |
|---|---|---:|---:|---:|
{rows_text}

输出目录：`data/tokenized/`
{ENCODING_SECTION_END}"""


def update_results_document(
    report_path: Path = RESULTS_PATH,
    output_directory: Path = TOKENIZED_DATA_DIRECTORY,
) -> None:
    """只替换结果文档中的(d)部分，保留(a-c)实验数字。"""

    if not report_path.is_file():
        raise FileNotFoundError(
            f"找不到 {report_path}，请先运行 problem_2_7_analyze_tokenizer.py。"
        )

    report = report_path.read_text(encoding="utf-8")
    start_index = report.find(ENCODING_SECTION_START)
    end_index = report.find(ENCODING_SECTION_END)
    if start_index == -1 or end_index == -1:
        raise ValueError(f"{report_path} 缺少 Problem 2.7(d) 区域标记。")

    end_index += len(ENCODING_SECTION_END)
    updated_report = (
        report[:start_index]
        + build_encoding_section(output_directory)
        + report[end_index:]
    )
    report_path.write_text(updated_report, encoding="utf-8")


def run_encoding(
    dataset_name: str,
    overwrite: bool,
    buffer_size: int,
    workers: int,
) -> None:
    """编码一个或全部数据集，每完成一个文件就更新结果文档。"""

    if dataset_name == "all":
        selected_datasets = DATASETS.items()
    else:
        selected_datasets = [(dataset_name, DATASETS[dataset_name])]

    for selected_name, config in selected_datasets:
        encode_dataset(
            dataset_name=selected_name,
            config=config,
            output_directory=TOKENIZED_DATA_DIRECTORY,
            overwrite=overwrite,
            buffer_size=buffer_size,
            workers=workers,
        )
        update_results_document()


def main() -> None:
    """读取命令行参数并启动长时间编码任务。"""

    parser = argparse.ArgumentParser(
        description="运行 Assignment 1 Problem 2.7(d) 全量数据编码",
    )
    parser.add_argument(
        "--dataset",
        choices=[*DATASETS, "all"],
        default="all",
        help="要编码的数据集（默认：all）",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=1_000_000,
        help="每批写入多少个 token（默认：1000000）",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允许覆盖已有编码文件",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"并行编码进程数（默认：{DEFAULT_WORKERS}）",
    )
    arguments = parser.parse_args()

    if arguments.buffer_size <= 0:
        parser.error("--buffer-size 必须大于 0")
    if arguments.workers <= 0:
        parser.error("--workers 必须大于 0")

    run_encoding(
        dataset_name=arguments.dataset,
        overwrite=arguments.overwrite,
        buffer_size=arguments.buffer_size,
        workers=arguments.workers,
    )


if __name__ == "__main__":
    main()
