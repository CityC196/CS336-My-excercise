"""Assignment 1, Problem 2.5：在 Modal 上训练 OpenWebText BPE。

第一次运行时，上传本地数据并训练：
    uv run modal run scripts/problem_2_5_train_openwebtext_modal.py --upload

数据已经上传到 Modal Volume 后，只运行训练：
    uv run modal run scripts/problem_2_5_train_openwebtext_modal.py
"""

import pickle
import resource
import time
from pathlib import Path

import modal


# OpenWebText 使用独立的 Modal App 名称，方便和 TinyStories 任务区分。
APP_NAME = "cs336-bpe-openwebtext-training"
VOLUME_NAME = "cs336-workspace"

# PDF 要求 OpenWebText tokenizer 的最大词表大小为 32,000。
VOCAB_SIZE = 32000
SPECIAL_TOKENS = ["<|endoftext|>"]

# 本地文件、Volume 文件和远程容器文件是三个不同的位置。
DEFAULT_LOCAL_DATA_PATH = "data/owt_train.txt"
VOLUME_INPUT_PATH = "/datasets/openwebtext/owt_train.txt"
REMOTE_INPUT_PATH = "/data/datasets/openwebtext/owt_train.txt"

# OWT 结果放在 openwebtext 子目录，不会覆盖 TinyStories 结果。
VOLUME_OUTPUT_DIRECTORY = "/assignments/assignment1/problem_2_5_openwebtext/final"
REMOTE_OUTPUT_DIRECTORY = "/data/assignments/assignment1/problem_2_5_openwebtext/final"


app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("regex>=2026.3.32")
    .add_local_python_source("cs336_basics")
)


def build_training_summary(
    elapsed_seconds: float,
    peak_memory_mb: float,
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
) -> str:
    """把课程要求检查的训练结果整理成一段文本。"""

    longest_token = max(vocab.values(), key=len)
    longest_token_text = longest_token.decode("utf-8", errors="replace")

    return (
        "数据集：OpenWebText train\n"
        f"训练时间：{elapsed_seconds} 秒\n"
        f"峰值内存：{peak_memory_mb} MB\n"
        f"词表大小：{len(vocab)}\n"
        f"merge 数量：{len(merges)}\n"
        f"最长 token：{longest_token!r}\n"
        f"最长 token 的 UTF-8 显示：{longest_token_text}\n"
        f"最长 token 字节数：{len(longest_token)}\n"
    )


@app.function(
    image=image,
    cpu=2.0,
    memory=95000,
    timeout=12 * 60 * 60,
    volumes={"/data": volume},
)
def train_on_modal() -> str:
    """读取 Volume 中的 OWT 训练集，训练 BPE，并保存结果。"""

    from cs336_basics.bpe import train_bpe

    input_path = Path(REMOTE_INPUT_PATH)
    output_directory = Path(REMOTE_OUTPUT_DIRECTORY)

    if not input_path.is_file():
        raise FileNotFoundError(
            "Modal Volume 中没有 OpenWebText 训练集。"
            "第一次运行请使用 --upload。"
        )

    output_directory.mkdir(parents=True, exist_ok=True)

    print("开始训练 OpenWebText BPE……")
    print("输入文件：", input_path)
    print("目标词表大小：", VOCAB_SIZE)
    print("特殊 token：", SPECIAL_TOKENS)

    start_time = time.perf_counter()

    vocab, merges = train_bpe(
        input_path=str(input_path),
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
    )

    elapsed_seconds = time.perf_counter() - start_time

    # Modal 远程容器使用 Linux，ru_maxrss 的单位是 KB；除以 1024 得到 MB。
    peak_memory_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    vocab_path = output_directory / "openwebtext_vocab.pkl"
    merges_path = output_directory / "openwebtext_merges.pkl"
    summary_path = output_directory / "training_summary.txt"

    with vocab_path.open("wb") as output_file:
        pickle.dump(vocab, output_file)

    with merges_path.open("wb") as output_file:
        pickle.dump(merges, output_file)

    summary = build_training_summary(
        elapsed_seconds=elapsed_seconds,
        peak_memory_mb=peak_memory_mb,
        vocab=vocab,
        merges=merges,
    )
    summary_path.write_text(summary, encoding="utf-8")

    # 主动提交，保证远程函数结束后结果仍保存在 Modal Volume 中。
    volume.commit()

    print("OpenWebText BPE 训练完成。")
    print(summary)
    print("结果目录：", output_directory)

    return summary


@app.local_entrypoint()
def main(
    upload: bool = False,
    local_data_path: str = DEFAULT_LOCAL_DATA_PATH,
) -> None:
    """按需上传本地 OWT 数据，然后启动 Modal 远程训练。"""

    if upload:
        local_path = Path(local_data_path)

        if not local_path.is_file():
            raise FileNotFoundError(f"找不到本地 OpenWebText 训练集：{local_path}")

        print("正在上传 OpenWebText 训练集：", local_path)
        print("文件较大，上传可能需要较长时间。")

        with volume.batch_upload() as upload_batch:
            upload_batch.put_file(str(local_path), VOLUME_INPUT_PATH)

        print("OpenWebText 训练集上传完成。")

    print("正在启动 Modal 远程训练……")
    summary = train_on_modal.remote()

    print("远程任务结束，训练摘要：")
    print(summary)
    print("查看远程结果：")
    print(
        "uv run modal volume ls cs336-workspace "
        f"{VOLUME_OUTPUT_DIRECTORY}"
    )
    print("下载前先创建本地目录：")
    print("mkdir -p ./outputs/problem_2_5_openwebtext")
    print("下载 OpenWebText 结果：")
    print(
        "uv run modal volume get cs336-workspace "
        f"{VOLUME_OUTPUT_DIRECTORY} ./outputs/problem_2_5_openwebtext"
    )
