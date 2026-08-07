"""Assignment 1, Problem 2.5：在 Modal 上训练 TinyStories BPE。

第一次上传完整训练集并正式训练：
    uv run modal run scripts/problem_2_5_train_tinystories_modal.py --upload

第一次上传 validation 集并进行性能分析：
    uv run modal run scripts/problem_2_5_train_tinystories_modal.py \
        --upload-validation --profile
"""

import cProfile
import io
import pickle
import pstats
import resource
import time
from pathlib import Path

import modal


APP_NAME = "cs336-bpe-training"
VOLUME_NAME = "cs336-workspace"

# 完整训练集：用于生成最终 vocab、merges 和正式训练时间。
DEFAULT_LOCAL_DATA_PATH = "data/TinyStoriesV2-GPT4-train.txt"
VOLUME_INPUT_PATH = "/datasets/tinystories/TinyStoriesV2-GPT4-train.txt"
REMOTE_INPUT_PATH = "/data/datasets/tinystories/TinyStoriesV2-GPT4-train.txt"

# Validation 集：数据更小，专门用于 cProfile 性能分析。
DEFAULT_LOCAL_VALIDATION_DATA_PATH = "data/TinyStoriesV2-GPT4-valid.txt"
VOLUME_VALIDATION_INPUT_PATH = "/datasets/tinystories/TinyStoriesV2-GPT4-valid.txt"
REMOTE_VALIDATION_INPUT_PATH = "/data/datasets/tinystories/TinyStoriesV2-GPT4-valid.txt"

# 正式结果和性能分析结果分开保存，避免相互覆盖。
REMOTE_OUTPUT_DIRECTORY = "/data/assignments/assignment1/problem_2_5_tinystories/final"
PROFILE_OUTPUT_DIRECTORY = (
    "/data/assignments/assignment1/problem_2_5_tinystories/profile_validation"
)


app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("regex>=2026.3.32")
    .add_local_python_source("cs336_basics")
)


@app.function(
    image=image,
    cpu=2.0,
    memory=30000,
    timeout=30 * 60,
    volumes={"/data": volume},
)
def train_on_modal(profile: bool = False) -> dict[str, bool | int | float | str]:
    """执行正式训练，或者在 validation 集上执行带 cProfile 的训练。"""

    from cs336_basics.bpe import train_bpe

    if profile:
        input_path = Path(REMOTE_VALIDATION_INPUT_PATH)
        output_directory = Path(PROFILE_OUTPUT_DIRECTORY)
        dataset_name = "TinyStories validation"
        missing_data_hint = "第一次性能分析请使用 --upload-validation --profile。"
    else:
        input_path = Path(REMOTE_INPUT_PATH)
        output_directory = Path(REMOTE_OUTPUT_DIRECTORY)
        dataset_name = "TinyStories train"
        missing_data_hint = "第一次正式训练请使用 --upload。"

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Modal Volume 中没有 {dataset_name} 数据。{missing_data_hint}"
        )

    output_directory.mkdir(parents=True, exist_ok=True)

    print("开始训练 BPE……")
    print("数据集：", dataset_name)
    print("性能分析：", "已启用" if profile else "未启用")

    profiler = None
    start_time = time.perf_counter()

    if profile:
        profiler = cProfile.Profile()
        profiler.enable()
        try:
            vocab, merges = train_bpe(
                input_path=str(input_path),
                vocab_size=10000,
                special_tokens=["<|endoftext|>"],
            )
        finally:
            profiler.disable()
    else:
        # 正式训练不启用 cProfile，避免约 2.5 倍的性能开销。
        vocab, merges = train_bpe(
            input_path=str(input_path),
            vocab_size=10000,
            special_tokens=["<|endoftext|>"],
        )

    elapsed_seconds = time.perf_counter() - start_time
    peak_memory_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    longest_token = max(vocab.values(), key=len)
    longest_token_text = longest_token.decode("utf-8", errors="replace")

    vocab_path = output_directory / "tinystories_vocab.pkl"
    merges_path = output_directory / "tinystories_merges.pkl"
    training_summary_path = output_directory / "training_summary.txt"

    with vocab_path.open("wb") as output_file:
        pickle.dump(vocab, output_file)

    with merges_path.open("wb") as output_file:
        pickle.dump(merges, output_file)

    profile_status = "已启用（训练时间包含 cProfile 开销）" if profile else "未启用"
    training_summary = (
        f"数据集：{dataset_name}\n"
        f"性能分析：{profile_status}\n"
        f"训练时间：{elapsed_seconds} 秒\n"
        f"峰值内存：{peak_memory_mb} MB\n"
        f"词表大小：{len(vocab)}\n"
        f"merge 数量：{len(merges)}\n"
        f"最长 token：{longest_token!r}\n"
        f"最长 token 的 UTF-8 显示：{longest_token_text}\n"
        f"最长 token 字节数：{len(longest_token)}\n"
    )
    training_summary_path.write_text(training_summary, encoding="utf-8")

    if profiler is not None:
        profile_path = output_directory / "bpe_profile.prof"
        profile_summary_path = output_directory / "profile_summary.txt"

        profiler.dump_stats(str(profile_path))

        profile_text_stream = io.StringIO()
        profile_statistics = pstats.Stats(profiler, stream=profile_text_stream)
        profile_statistics.strip_dirs().sort_stats("cumulative").print_stats(30)
        profile_summary = profile_text_stream.getvalue()
        profile_summary_path.write_text(profile_summary, encoding="utf-8")

        print("性能分析（按累计耗时排序的前 30 项）：")
        print(profile_summary)

    volume.commit()

    print("训练完成")
    print(training_summary)
    print("结果目录：", output_directory)

    return {
        "profile_enabled": profile,
        "elapsed_seconds": elapsed_seconds,
        "peak_memory_mb": peak_memory_mb,
        "vocab_size": len(vocab),
        "merge_count": len(merges),
        "longest_token": repr(longest_token),
        "longest_token_text": longest_token_text,
        "longest_token_bytes": len(longest_token),
    }


@app.local_entrypoint()
def main(
    upload: bool = False,
    upload_validation: bool = False,
    profile: bool = False,
    local_data_path: str = DEFAULT_LOCAL_DATA_PATH,
    local_validation_data_path: str = DEFAULT_LOCAL_VALIDATION_DATA_PATH,
) -> None:
    """在本地上传需要的数据，然后启动对应的远程训练模式。"""

    if upload:
        local_path = Path(local_data_path)
        if not local_path.is_file():
            raise FileNotFoundError(f"找不到本地训练数据：{local_path}")

        print("正在上传完整训练集：", local_path)
        with volume.batch_upload() as upload_batch:
            upload_batch.put_file(str(local_path), VOLUME_INPUT_PATH)
        print("完整训练集上传完成。")

    if upload_validation:
        local_validation_path = Path(local_validation_data_path)
        if not local_validation_path.is_file():
            raise FileNotFoundError(
                f"找不到本地 validation 数据：{local_validation_path}"
            )

        print("正在上传 validation 集：", local_validation_path)
        with volume.batch_upload() as upload_batch:
            upload_batch.put_file(
                str(local_validation_path),
                VOLUME_VALIDATION_INPUT_PATH,
            )
        print("Validation 集上传完成。")

    print("正在启动 Modal 远程训练……")
    summary = train_on_modal.remote(profile)

    print("远程任务结束，摘要：", summary)
    print("下载全部结果请运行：")
    print(
        "uv run modal volume get cs336-workspace "
        "/assignments/assignment1/problem_2_5_tinystories "
        "./outputs/problem_2_5_tinystories"
    )
