# Assignment 1 Tokenizer 部分文件结构

这份文档只说明作业要求的代码、调用关系和产物位置。官方提供的源码、测试、fixtures 和 snapshots 都保持原来的文件名。

## 1. 题号与文件对应关系

| 题目 | 文件 | 作用 |
|---|---|---|
| Problem 2.4（15 分） | `cs336_basics/bpe.py` | 实现 `train_bpe`，从语料训练 `vocab` 和 `merges` |
| Problem 2.5 | `scripts/problem_2_5_train_tinystories_modal.py` | 在 TinyStories 上训练 10K BPE，并在 validation 上做 profile |
| Problem 2.5 | `scripts/problem_2_5_train_openwebtext_modal.py` | 在 OpenWebText 上训练 32K BPE |
| Problem 2.6（15 分） | `cs336_basics/tokenizer.py` | 实现 `Tokenizer` 的 encode、decode 和 encode_iterable |
| Problem 2.7(a-c) | `scripts/problem_2_7_analyze_tokenizer.py` | 抽样、压缩率、跨域和吞吐量实验 |
| Problem 2.7(d) | `scripts/problem_2_7_encode_datasets.py` | 把四个完整数据集流式编码为 uint16 |
| Problem 2.7 结果 | `docs/problem_2_7_results.md` | 集中保存全部实验数字和文字回答 |
| 官方测试连接 | `tests/adapters.py` | 让官方测试调用 `train_bpe` 和 `Tokenizer` |

`cs336_basics/` 存放可以被其他 Python 文件 `import` 的算法和类。`scripts/` 存放需要从命令行直接运行的训练或实验流程；脚本本身不重复实现 BPE，而是调用 `cs336_basics/` 中的代码。

## 2. 完整调用结构

```text
官方 BPE 测试
tests/test_train_bpe.py
  -> tests/adapters.py::run_train_bpe
  -> cs336_basics/bpe.py::train_bpe

Problem 2.5 训练脚本
scripts/problem_2_5_train_*_modal.py
  -> cs336_basics/bpe.py::train_bpe
  -> 保存 vocab.pkl、merges.pkl、training_summary.txt

官方 Tokenizer 测试
tests/test_tokenizer.py
  -> tests/adapters.py::get_tokenizer
  -> cs336_basics/tokenizer.py::Tokenizer

Problem 2.7(a-c) 分析脚本
scripts/problem_2_7_analyze_tokenizer.py
  -> Tokenizer.from_files(vocab.pkl, merges.pkl)
  -> Tokenizer.encode
  -> docs/problem_2_7_results.md

Problem 2.7(d) 编码脚本
scripts/problem_2_7_encode_datasets.py
  -> Tokenizer.from_files(vocab.pkl, merges.pkl)
  -> Tokenizer.encode_iterable
  -> data/tokenized/*.uint16.bin
  -> 更新 docs/problem_2_7_results.md 的(d)部分
```

`vocab.pkl` 保存 `dict[int, bytes]`，也就是 token ID 到 token 字节串的映射。`merges.pkl` 保存有顺序的 `list[tuple[bytes, bytes]]`；列表顺序就是 encode 应用 BPE merge 的优先级。

## 3. 目录结构

```text
assignment1-basics-main/
├── cs336_basics/
│   ├── bpe.py
│   ├── tokenizer.py
│   └── pretokenization_example.py
├── scripts/
│   ├── problem_2_5_train_tinystories_modal.py
│   ├── problem_2_5_train_openwebtext_modal.py
│   ├── problem_2_7_analyze_tokenizer.py
│   └── problem_2_7_encode_datasets.py
├── tests/                         # 只保留课程提供的官方测试
├── data/
│   ├── TinyStoriesV2-GPT4-train.txt
│   ├── TinyStoriesV2-GPT4-valid.txt
│   ├── owt_train.txt
│   ├── owt_valid.txt
│   └── tokenized/                 # Problem 2.7(d) 的四个 uint16 文件
├── docs/
│   └── problem_2_7_results.md     # Problem 2.7 唯一结果文档
└── outputs/
    ├── problem_2_5_tinystories/
    │   ├── final/
    │   └── profile_validation/
    ├── problem_2_5_openwebtext/
    │   └── final/
    └── legacy/                    # 旧版本参数，仅归档，不供新脚本调用
```

## 4. 运行方式

只运行 tokenizer 部分的官方测试：

```bash
uv run pytest tests/test_train_bpe.py tests/test_tokenizer.py -v
```

TinyStories 正式训练：

```bash
uv run modal run scripts/problem_2_5_train_tinystories_modal.py --upload
```

TinyStories validation 性能分析：

```bash
uv run modal run scripts/problem_2_5_train_tinystories_modal.py --upload-validation --profile
```

OpenWebText 正式训练：

```bash
uv run modal run scripts/problem_2_5_train_openwebtext_modal.py --upload
```

Problem 2.7(a–c) 实验：

```bash
uv run python scripts/problem_2_7_analyze_tokenizer.py
```

结果只写入 `docs/problem_2_7_results.md`，不保存抽样文档或 JSON 中间文件。

Problem 2.7(d) 编码全部四个数据集：

```bash
uv run python scripts/problem_2_7_encode_datasets.py --dataset all
```

编码结果写入 `data/tokenized/`，每完成一个数据集就更新结果文档中的 2.7(d) 表格。为了防止误覆盖，
已有结果存在时脚本会停止；确认要重做时再增加 `--overwrite`。

生成的文件可以直接按下列方式读取，不需要一次全部载入内存：

```python
import numpy as np

token_ids = np.memmap(
    "data/tokenized/problem_2_7_tinystories_train.uint16.bin",
    dtype=np.uint16,
    mode="r",
)
```

正式训练和四个大文件编码都可能运行很久。整理目录或运行测试时，不会自动启动这些任务。
