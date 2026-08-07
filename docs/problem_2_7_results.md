# Assignment 1 — Problem 2.7 Tokenizer Experiments

## 实验设置

- 随机种子：`336`
- 每个 validation 数据集抽取 `10` 个非空文档
- 压缩率定义：原文本 UTF-8 字节数除以编码后的 token 数，即 `bytes/token`

## (a) 同域压缩率

| 数据集 | Tokenizer | UTF-8 字节数 | Token 数 | bytes/token |
|---|---|---:|---:|---:|
| TinyStories | TinyStories 10K | 8733 | 2106 | 4.1467 |
| OpenWebText | OpenWebText 32K | 32949 | 7389 | 4.4592 |

`bytes/token` 越大，表示一个 token 平均表示的原始字节越多，压缩效率越高。本次抽样中，
OpenWebText 32K tokenizer 的压缩率更高。主要原因是它的词表更大，而且训练语料覆盖的文本类型更广。

## (b) 跨域压缩率

| 数据集 | Tokenizer | UTF-8 字节数 | Token 数 | bytes/token |
|---|---|---:|---:|---:|
| OpenWebText | OpenWebText 32K | 32949 | 7389 | 4.4592 |
| OpenWebText | TinyStories 10K | 32949 | 9866 | 3.3397 |

TinyStories tokenizer 编码 OpenWebText 时产生更多 token，`bytes/token` 明显下降。这说明除了词表大小，
tokenizer 的训练语料领域也会直接影响压缩效率。

## (c) 编码吞吐量

- OpenWebText 样本吞吐量：`919079.79 bytes/s`
- 编码 825GB The Pile 的估计时间：`897636.98` 秒
- 换算结果：约 `249.34` 小时，即 `10.39` 天

吞吐量会受到 CPU、后台负载和样本文档长度影响，所以这里是当前机器上的估计值。

## (d) 全量数据编码

<!-- PROBLEM_2_7_D_START -->
四个文件均使用对应语料训练得到的 tokenizer，并以原始 `uint16` 顺序保存，可由 `np.memmap` 直接读取。

| 数据集 | 状态 | 原始字节数 | Token 数 | bytes/token |
|---|---|---:|---:|---:|
| tinystories_valid | 完成 | 22,502,601 | 5,461,210 | 4.1204 |
| openwebtext_valid | 完成 | 289,998,753 | 66,401,098 | 4.3674 |
| tinystories_train | 完成 | 2,227,753,162 | 540,796,778 | 4.1194 |
| openwebtext_train | 完成 | 11,920,511,059 | 2,727,120,452 | 4.3711 |

输出目录：`data/tokenized/`
<!-- PROBLEM_2_7_D_END -->
