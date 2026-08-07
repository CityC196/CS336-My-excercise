"""Assignment 1, Problem 2.7(a-c)：运行 tokenizer 实验并写作业结果。

运行方式：

    uv run python scripts/problem_2_7_analyze_tokenizer.py

本脚本只生成 ``docs/problem_2_7_results.md``，不会保存抽样文本或 JSON 中间文件。
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

from cs336_basics.tokenizer import Tokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPECIAL_TOKEN = "<|endoftext|>"
SAMPLE_SIZE = 10
RANDOM_SEED = 336
PILE_SIZE_IN_BYTES = 825_000_000_000

TINYSTORIES_VALIDATION_PATH = PROJECT_ROOT / "data" / "TinyStoriesV2-GPT4-valid.txt"
OPENWEBTEXT_VALIDATION_PATH = PROJECT_ROOT / "data" / "owt_valid.txt"

TINYSTORIES_TOKENIZER_DIRECTORY = (
    PROJECT_ROOT / "outputs" / "problem_2_5_tinystories" / "final"
)
OPENWEBTEXT_TOKENIZER_DIRECTORY = (
    PROJECT_ROOT / "outputs" / "problem_2_5_openwebtext" / "final"
)

RESULTS_PATH = PROJECT_ROOT / "docs" / "problem_2_7_results.md"
ENCODING_SECTION_START = "<!-- PROBLEM_2_7_D_START -->"
ENCODING_SECTION_END = "<!-- PROBLEM_2_7_D_END -->"


def load_tokenizer(
    tokenizer_directory: Path,
    vocab_filename: str,
    merges_filename: str,
) -> Tokenizer:
    """从 Problem 2.5 训练得到的 vocab、merges 构造 Tokenizer。"""

    vocab_path = tokenizer_directory / vocab_filename
    merges_path = tokenizer_directory / merges_filename

    if not vocab_path.is_file() or not merges_path.is_file():
        raise FileNotFoundError(
            "找不到 tokenizer 参数，请检查以下文件：\n"
            f"  {vocab_path}\n"
            f"  {merges_path}"
        )

    return Tokenizer.from_files(
        vocab_filepath=vocab_path,
        merges_filepath=merges_path,
        special_tokens=[SPECIAL_TOKEN],
    )


def sample_documents(
    input_path: Path,
    sample_size: int = SAMPLE_SIZE,
    seed: int = RANDOM_SEED,
) -> list[str]:
    """均匀抽取文档，同时避免把整个 validation 文件读入内存。"""

    if not input_path.is_file():
        raise FileNotFoundError(f"找不到 validation 数据：{input_path}")

    separator = SPECIAL_TOKEN.encode("utf-8")
    random_generator = random.Random(seed)
    sampled_documents: list[bytes] = []
    nonempty_document_count = 0
    unfinished_document = b""

    with input_path.open("rb") as input_file:
        while chunk := input_file.read(1024 * 1024):
            documents = (unfinished_document + chunk).split(separator)
            unfinished_document = documents.pop()

            for document in documents:
                if not document.strip():
                    continue

                nonempty_document_count += 1
                if len(sampled_documents) < sample_size:
                    sampled_documents.append(document)
                    continue

                replacement_index = random_generator.randrange(nonempty_document_count)
                if replacement_index < sample_size:
                    sampled_documents[replacement_index] = document

    if unfinished_document.strip():
        nonempty_document_count += 1
        if len(sampled_documents) < sample_size:
            sampled_documents.append(unfinished_document)
        else:
            replacement_index = random_generator.randrange(nonempty_document_count)
            if replacement_index < sample_size:
                sampled_documents[replacement_index] = unfinished_document

    if len(sampled_documents) < sample_size:
        raise ValueError(
            f"{input_path.name} 只有 {len(sampled_documents)} 个非空文档，"
            f"不能抽取 {sample_size} 个。"
        )

    return [document.decode("utf-8", errors="replace") for document in sampled_documents]


def measure_compression(
    tokenizer: Tokenizer,
    documents: list[str],
) -> dict[str, float | int]:
    """计算 UTF-8 字节数、token 数和压缩率 bytes/token。"""

    byte_count = 0
    token_count = 0

    for document in documents:
        byte_count += len(document.encode("utf-8"))
        token_count += len(tokenizer.encode(document))

    return {
        "byte_count": byte_count,
        "token_count": token_count,
        "bytes_per_token": byte_count / token_count,
    }


def measure_throughput(
    tokenizer: Tokenizer,
    documents: list[str],
    minimum_seconds: float,
) -> dict[str, float]:
    """重复编码样本文档，测量每秒处理的 UTF-8 字节数。"""

    # 先预热，避免首次调用的固定开销影响计时。
    for document in documents:
        tokenizer.encode(document)

    bytes_per_round = sum(len(document.encode("utf-8")) for document in documents)
    processed_bytes = 0
    start_time = time.perf_counter()
    elapsed_seconds = 0.0

    while elapsed_seconds < minimum_seconds:
        for document in documents:
            tokenizer.encode(document)
        processed_bytes += bytes_per_round
        elapsed_seconds = time.perf_counter() - start_time

    bytes_per_second = processed_bytes / elapsed_seconds
    estimated_pile_seconds = PILE_SIZE_IN_BYTES / bytes_per_second

    return {
        "bytes_per_second": bytes_per_second,
        "estimated_pile_seconds": estimated_pile_seconds,
        "estimated_pile_hours": estimated_pile_seconds / 3600,
        "estimated_pile_days": estimated_pile_seconds / 86400,
    }


def get_existing_encoding_section(report_path: Path) -> str:
    """重新运行(a-c)时保留已经写入 Markdown 的(d)编码结果。"""

    if report_path.is_file():
        old_report = report_path.read_text(encoding="utf-8")
        start_index = old_report.find(ENCODING_SECTION_START)
        end_index = old_report.find(ENCODING_SECTION_END)

        if start_index != -1 and end_index != -1:
            end_index += len(ENCODING_SECTION_END)
            return old_report[start_index:end_index]

    return f"""{ENCODING_SECTION_START}
尚未运行全量编码。完成后这里会列出四个二进制文件的数据检查结果。
{ENCODING_SECTION_END}"""


def write_results_document(
    report_path: Path,
    tiny_metrics: dict[str, float | int],
    owt_metrics: dict[str, float | int],
    cross_metrics: dict[str, float | int],
    throughput_metrics: dict[str, float],
) -> None:
    """把 Problem 2.7 的实验数字和文字回答集中写入一个 Markdown。"""

    encoding_section = get_existing_encoding_section(report_path)
    report = f"""# Assignment 1 — Problem 2.7 Tokenizer Experiments

## 实验设置

- 随机种子：`{RANDOM_SEED}`
- 每个 validation 数据集抽取 `{SAMPLE_SIZE}` 个非空文档
- 压缩率定义：原文本 UTF-8 字节数除以编码后的 token 数，即 `bytes/token`

## (a) 同域压缩率

| 数据集 | Tokenizer | UTF-8 字节数 | Token 数 | bytes/token |
|---|---|---:|---:|---:|
| TinyStories | TinyStories 10K | {tiny_metrics['byte_count']} | {tiny_metrics['token_count']} | {tiny_metrics['bytes_per_token']:.4f} |
| OpenWebText | OpenWebText 32K | {owt_metrics['byte_count']} | {owt_metrics['token_count']} | {owt_metrics['bytes_per_token']:.4f} |

`bytes/token` 越大，表示一个 token 平均表示的原始字节越多，压缩效率越高。本次抽样中，
OpenWebText 32K tokenizer 的压缩率更高。主要原因是它的词表更大，而且训练语料覆盖的文本类型更广。

## (b) 跨域压缩率

| 数据集 | Tokenizer | UTF-8 字节数 | Token 数 | bytes/token |
|---|---|---:|---:|---:|
| OpenWebText | OpenWebText 32K | {owt_metrics['byte_count']} | {owt_metrics['token_count']} | {owt_metrics['bytes_per_token']:.4f} |
| OpenWebText | TinyStories 10K | {cross_metrics['byte_count']} | {cross_metrics['token_count']} | {cross_metrics['bytes_per_token']:.4f} |

TinyStories tokenizer 编码 OpenWebText 时产生更多 token，`bytes/token` 明显下降。这说明除了词表大小，
tokenizer 的训练语料领域也会直接影响压缩效率。

## (c) 编码吞吐量

- OpenWebText 样本吞吐量：`{throughput_metrics['bytes_per_second']:.2f} bytes/s`
- 编码 825GB The Pile 的估计时间：`{throughput_metrics['estimated_pile_seconds']:.2f}` 秒
- 换算结果：约 `{throughput_metrics['estimated_pile_hours']:.2f}` 小时，即 `{throughput_metrics['estimated_pile_days']:.2f}` 天

吞吐量会受到 CPU、后台负载和样本文档长度影响，所以这里是当前机器上的估计值。

## (d) 全量数据编码

{encoding_section}
"""

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print("Problem 2.7 结果已写入：", report_path)


def run_analysis(minimum_seconds: float) -> None:
    """执行 Problem 2.7(a-c) 的全部实验。"""

    tiny_tokenizer = load_tokenizer(
        tokenizer_directory=TINYSTORIES_TOKENIZER_DIRECTORY,
        vocab_filename="tinystories_vocab.pkl",
        merges_filename="tinystories_merges.pkl",
    )
    owt_tokenizer = load_tokenizer(
        tokenizer_directory=OPENWEBTEXT_TOKENIZER_DIRECTORY,
        vocab_filename="openwebtext_vocab.pkl",
        merges_filename="openwebtext_merges.pkl",
    )

    print("正在从 TinyStories validation 抽取 10 个文档……")
    tiny_documents = sample_documents(TINYSTORIES_VALIDATION_PATH)
    print("正在从 OpenWebText validation 抽取 10 个文档……")
    owt_documents = sample_documents(OPENWEBTEXT_VALIDATION_PATH)

    print("正在计算压缩率……")
    tiny_metrics = measure_compression(tiny_tokenizer, tiny_documents)
    owt_metrics = measure_compression(owt_tokenizer, owt_documents)
    cross_metrics = measure_compression(tiny_tokenizer, owt_documents)

    print("正在测量 OpenWebText tokenizer 吞吐量……")
    throughput_metrics = measure_throughput(
        tokenizer=owt_tokenizer,
        documents=owt_documents,
        minimum_seconds=minimum_seconds,
    )

    write_results_document(
        report_path=RESULTS_PATH,
        tiny_metrics=tiny_metrics,
        owt_metrics=owt_metrics,
        cross_metrics=cross_metrics,
        throughput_metrics=throughput_metrics,
    )


def main() -> None:
    """读取命令行参数并运行实验。"""

    parser = argparse.ArgumentParser(
        description="运行 Assignment 1 Problem 2.7(a-c) tokenizer 实验",
    )
    parser.add_argument(
        "--minimum-seconds",
        type=float,
        default=3.0,
        help="吞吐量测量至少运行多少秒（默认：3.0）",
    )
    arguments = parser.parse_args()

    if arguments.minimum_seconds <= 0:
        parser.error("--minimum-seconds 必须大于 0")

    run_analysis(minimum_seconds=arguments.minimum_seconds)


if __name__ == "__main__":
    main()
