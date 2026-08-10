# CS336 Assignment 1: Basics

这是 Stanford CS336 *Language Modeling from Scratch* 的 Assignment 1 练习仓库，内容包括 byte-level BPE、Tokenizer、tokenizer experiments，以及 Transformer 的基础神经网络模块。

作业说明：[Stanford CS336 Assignment 1: Basics](https://github.com/stanford-cs336/assignment1-basics/blob/main/cs336_assignment1_basics.pdf)

## 题号与代码位置

| 题目 | 状态 | 内容 | 主要实现 | 测试、adapter 或实验结果 |
| --- | --- | --- | --- | --- |
| Problem 2.4 | 已完成 | 训练 byte-level BPE tokenizer | [`cs336_basics/bpe.py`](./cs336_basics/bpe.py) | [`tests/test_train_bpe.py`](./tests/test_train_bpe.py)、`tests/adapters.py::run_train_bpe` |
| Problem 2.5 | 已完成 | 在 TinyStories 和 OpenWebText 上训练 tokenizer | [`scripts/problem_2_5_train_tinystories_modal.py`](./scripts/problem_2_5_train_tinystories_modal.py)、[`scripts/problem_2_5_train_openwebtext_modal.py`](./scripts/problem_2_5_train_openwebtext_modal.py) | 训练脚本与本地输出目录 |
| Problem 2.6 | 已完成 | `Tokenizer.encode`、`decode` 和 `encode_iterable` | [`cs336_basics/tokenizer.py`](./cs336_basics/tokenizer.py) | [`tests/test_tokenizer.py`](./tests/test_tokenizer.py)、`tests/adapters.py::get_tokenizer` |
| Problem 2.7 | 已完成 | 压缩率、跨域、吞吐量和全量数据编码实验 | [`scripts/problem_2_7_analyze_tokenizer.py`](./scripts/problem_2_7_analyze_tokenizer.py)、[`scripts/problem_2_7_encode_datasets.py`](./scripts/problem_2_7_encode_datasets.py) | [`docs/problem_2_7_results.md`](./docs/problem_2_7_results.md) |
| Problem 3.3.2 | 已完成 | 无 bias 的 Linear 模块 | `cs336_basics/model.py::Linear` | `tests/adapters.py::run_linear`、`tests/test_model.py::test_linear` |
| Problem 3.3.3 | 已完成 | Embedding 查表模块 | `cs336_basics/model.py::Embedding` | `tests/adapters.py::run_embedding`、`tests/test_model.py::test_embedding` |
| Problem 3.4.1 | 已完成 | RMSNorm 归一化模块 | `cs336_basics/model.py::RMSNorm` | `tests/adapters.py::run_rmsnorm`、`tests/test_model.py::test_rmsnorm` |
| Problem 3.4.2 | 已完成 | SwiGLU 前馈网络 | `cs336_basics/model.py::SwiGLU` | `tests/adapters.py::run_swiglu`、`tests/test_model.py::test_swiglu` |

核心实现放在 `cs336_basics/`，独立训练和实验程序放在 `scripts/`，课程测试及其 adapter 放在 `tests/`。`docs/` 只保留需要提交或记录的作业答案与实验结果。

## 安装与测试

项目使用 Python 3.12、PyTorch 和 `uv`：

```bash
uv sync
```

测试 tokenizer 部分：

```bash
uv run pytest tests/test_train_bpe.py tests/test_tokenizer.py -v
```

测试 Linear、Embedding、RMSNorm 和 SwiGLU：

```bash
uv run pytest -k "test_linear or test_embedding or test_rmsnorm or test_swiglu" -v
```

## 数据说明

仓库不包含 TinyStories、OpenWebText、tokenized `.bin` 文件或本地训练输出。运行 Problem 2.5 和 Problem 2.7 前，请按照作业说明准备数据集、模型参数和本地输出目录。
