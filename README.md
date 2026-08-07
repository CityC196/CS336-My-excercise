# CS336 My Exercise

这是我的 CS336 Assignment 1 练习仓库。目前已经完成 byte-level BPE 训练、Tokenizer 实现，
以及 TinyStories/OpenWebText tokenizer experiments。

## 已完成内容

- Problem 2.4：训练 byte-level BPE tokenizer
- Problem 2.5：TinyStories 10K 和 OpenWebText 32K BPE 训练脚本
- Problem 2.6：`Tokenizer.encode`、`decode` 和 `encode_iterable`
- Problem 2.7：压缩率、跨域、吞吐量和全量数据编码实验

主要代码：

- `cs336_basics/bpe.py`
- `cs336_basics/tokenizer.py`
- `scripts/problem_2_5_train_tinystories_modal.py`
- `scripts/problem_2_5_train_openwebtext_modal.py`
- `scripts/problem_2_7_analyze_tokenizer.py`
- `scripts/problem_2_7_encode_datasets.py`

## 安装与测试

项目使用 Python 3.12 和 `uv`：

```bash
uv sync
uv run pytest tests/test_train_bpe.py tests/test_tokenizer.py -v
```

当前预期结果：`27 passed, 1 xfailed`。其中 xfail 是课程测试明确标记的
`Tokenizer.encode` 内存限制预期失败。

## 数据说明

仓库不包含 TinyStories、OpenWebText、tokenized `.bin` 文件或本地训练输出。
运行 Problem 2.5/2.7 前，需要按照 `docs/assignment1_tokenizer.md` 准备本地数据和训练参数。

Problem 2.7 的实验数字和文字结论保存在 `docs/problem_2_7_results.md`。
