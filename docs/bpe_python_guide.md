# 从 C++ 到 Python：用 Assignment 1 的 BPE 训练入门

这份文档面向“会 C++，基本不会 Python”的读者。建议你一边看
`cs336_basics/bpe.py`，一边阅读本文，并亲手修改小例子运行。

## 1. 先看最终成果

学习这个任务时，主要看下面两个代码位置：

- `cs336_basics/bpe.py`：真正的 BPE 训练代码。
- `tests/adapters.py`：课程测试与实现之间的转接函数。

对外函数是：

```python
def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
```

它可以粗略理解为下面的 C++ 声明：

```cpp
pair<map<int, Bytes>, vector<pair<Bytes, Bytes>>> train_bpe(
    Path input_path,
    int vocab_size,
    vector<string> special_tokens
);
```

运行课程测试：

```bash
uv run pytest tests/test_train_bpe.py -v
```

## 2. BPE 到底在做什么

BPE（Byte-Pair Encoding）的核心动作很简单：

1. 找到当前出现次数最多的相邻 token pair。
2. 把这一对合并成一个新 token。
3. 重复以上步骤，直到词表达到指定大小。

用文本 `aa aa` 手算。课程正则预分词后得到：

```text
aa
 aa
```

注意第二个 pre-token 带一个前导空格。初始时每个字节都是一个 token：

```text
[a, a]
[空格, a, a]
```

第一轮统计：

```text
(a, a)      出现 2 次
(空格, a)   出现 1 次
```

所以先合并 `(a, a)`：

```text
[aa]
[空格, aa]
```

第二轮只剩 `(空格, aa)`，于是继续合并。最终 merges 是：

```python
[(b"a", b"a"), (b" ", b"aa")]
```

新词表项依次是 `b"aa"` 和 `b" aa"`。

## 3. 完整数据流

程序处理数据的顺序是：

```text
UTF-8 文本文件
    ↓ 读取为 str
按 special token 切成普通文本片段
    ↓ 课程指定正则表达式
pre-token 及其出现次数
    ↓ str.encode("utf-8")
由单字节 bytes 组成的 token 序列
    ↓ 统计相邻 pair
pair_counts 和 pair_to_word_ids
    ↓ 反复选择、合并、局部更新
vocab 和 merges
```

为什么先做 pre-tokenization？如果直接对整个文件做 BPE，程序可能把不同单词之间的内容随意
合并。课程给出的正则先把单词、数字、标点和空白分成合理的小块。BPE merge 只能发生在同一个
pre-token 内部。

为什么统计 pre-token 的频率？语料中可能出现十万次 `the`。我们不需要保存十万份相同的
`[t, h, e]`，只保存一份，再记录 `frequency = 100000`。统计 pair 时乘上 frequency 即可。

## 4. Python 与 C++ 快速对照

### 4.1 函数、缩进和返回值

Python：

```python
def add(left: int, right: int) -> int:
    result = left + right
    return result
```

C++：

```cpp
int add(int left, int right) {
    int result = left + right;
    return result;
}
```

需要注意：

- Python 用 `def` 定义函数。
- `:` 表示下面将开始一个缩进代码块。
- Python 用缩进代替 C++ 的 `{}`，一般不写分号。
- `left: int` 和 `-> int` 是类型标注，帮助读者和检查工具理解代码。
- Python 默认不会在运行时严格强制类型标注。

### 4.2 `None`

`None` 表示“没有值”。在代码中：

```python
best_pair: Pair | None = None
```

`Pair | None` 表示变量可能是一个 `Pair`，也可能暂时没有值。

```python
if best_pair is None:
    break
```

判断 `None` 时通常使用 `is None`，不要写 `== None`。

### 4.3 四种必须认识的容器

| Python | 类似的 C++ 类型 | 特点 | 本任务中的用途 |
|---|---|---|---|
| `list` | `std::vector` | 有顺序、可修改 | token 序列、所有词、merge 列表 |
| `dict` | `std::unordered_map` | key 映射到 value | 词表、频率统计 |
| `set` | `std::unordered_set` | 元素不重复 | 一个 pair 涉及哪些 word ID |
| `tuple` | `std::tuple` / `std::pair` | 有顺序、不可修改 | pair、可以作为字典 key 的词序列 |

例子：

```python
numbers = [10, 20, 30]
counts = {"a": 2, "b": 1}
word_ids = {0, 3, 8}
pair = (b"a", b"b")
```

字典的 key 必须可哈希。`list` 可以修改，因此不能作为 key；`tuple` 不可修改，可以作为 key。
真正开始合并时又要修改 token 序列，所以程序把 tuple 转回 list：

```python
words.append(list(word))
```

## 5. `str` 和 `bytes` 是两种不同类型

这是本任务最容易混淆的地方。

### 5.1 `str` 表示人类文本

```python
text = "你"
print(type(text))
```

输出是 `<class 'str'>`。`str` 中保存的是 Unicode 字符。

### 5.2 `bytes` 表示原始字节

```python
encoded = "你".encode("utf-8")
print(encoded)
print(list(encoded))
```

输出类似：

```text
b'\xe4\xbd\xa0'
[228, 189, 160]
```

一个汉字在 UTF-8 中可能由多个字节组成。byte-level BPE 的初始 token 不是“一个字符”，
而是“一个字节”。因此 `你` 初始会变成三个 token：

```python
[b"\xe4", b"\xbd", b"\xa0"]
```

遍历 `bytes` 时得到的是整数。把某个整数重新变成单字节 `bytes`：

```python
for byte_value in encoded:
    one_byte = bytes([byte_value])
    print(one_byte)
```

拼接两个 bytes token：

```python
merged = b"a" + b"b"
assert merged == b"ab"
```

## 6. 代码逐函数讲解

### 6.1 `Pair` 类型别名

```python
Pair = tuple[bytes, bytes]
```

它只是给复杂类型起一个短名字。以后写 `Pair`，就是指两个 `bytes` 组成的 tuple。

### 6.2 `_count_pretokens`

职责：

1. 先按特殊 token 切开文本。
2. 丢掉特殊 token 自身。
3. 对普通文本运行课程正则。
4. 把每个结果转换为单字节序列。
5. 用普通 dict 统计重复次数。

假设文本是 `a<|endoftext|>a`，正确结果是两个独立的 `a`。如果只是删除特殊 token 再
预分词，会变成 `aa`，BPE 就可能错误地跨边界做 merge。

构造特殊 token 正则的关键代码：

```python
escaped_tokens = []

for token in special_tokens:
    escaped_tokens.append(regex.escape(token))

special_pattern = "|".join(escaped_tokens)
text_parts = regex.split(special_pattern, text)
```

- `regex.escape`：让 `<`、`|` 等字符被当作普通字符。
- `"|".join(...)`：用正则的“或者”符号连接多个特殊 token。
- `regex.split`：按特殊 token 切开文本，并且不把分隔符放进结果。

统计 pre-token 使用普通字典：

```python
pretoken = tuple(byte_tokens)
pretoken_counts[pretoken] = pretoken_counts.get(pretoken, 0) + 1
```

### 6.3 `_count_pairs_in_word`

职责：统计一个词内部所有相邻 pair 的次数。

```python
for index in range(len(word) - 1):
    pair = (word[index], word[index + 1])
    pair_counts[pair] = pair_counts.get(pair, 0) + 1
```

如果 `word` 有 4 个 token，就有 3 个相邻位置。`dict.get(pair, 0)` 表示 key 已存在时取原值，
不存在时使用默认值 0。

### 6.4 `_merge_pair_in_word`

职责：从左到右做非重叠 merge。

```text
while 还没有处理完：
    if 当前和下一个 token 是目标 pair：
        添加合并后的 token，前进 2 格
    else：
        原样添加当前 token，前进 1 格
```

这里使用 `while`，因为一次循环后可能前进 1 格，也可能前进 2 格。该函数创建并返回
`new_word`，不会修改传入的 `word`，这样更容易理解和测试。

### 6.5 `_choose_best_pair`

职责：实现课程规定的选择规则。

```python
if count > best_count:
    best_pair = pair
    best_count = count
elif count == best_count and best_pair is not None and pair > best_pair:
    best_pair = pair
```

第一优先级是频率；只有频率相等，才比较 pair 本身。Python 的 `bytes` 按字节顺序比较，
tuple 会先比较第一个元素，再在相等时比较第二个元素。

### 6.6 `_build_pair_statistics`

职责：在训练开始时建立两份统计：

```python
pair_counts[pair]
pair_to_word_ids[pair]
```

`pair_counts` 回答“这个 pair 在整个语料中出现多少次”；`pair_to_word_ids` 回答“这个 pair
存在哪些不同的 pre-token 中”。一个词内部的 pair 次数必须乘以该词在语料中的频率：

```python
weighted_count = occurrences * frequency
```

### 6.7 `_update_words_for_merge`

职责：选择最佳 pair 后，只更新包含它的 pre-token。对每个受影响的词做三步：

1. 从全局统计中减去旧 token 序列的贡献。
2. 执行 merge。
3. 把新 token 序列的贡献加回全局统计。

```python
affected_word_ids = list(pair_to_word_ids[pair_to_merge])
```

后面的循环会修改原 set，所以先用 `list(...)` 创建当前 ID 的副本。

### 6.8 `train_bpe`

主函数把前面的零件串起来。它先构造 256 个初始 byte token：

```python
vocab: dict[int, bytes] = {}
for byte_value in range(256):
    vocab[byte_value] = bytes([byte_value])
```

然后追加特殊 token。因为 ID 从 0 连续增长，当前 `len(vocab)` 正好是下一个 ID：

```python
for special_token in special_tokens:
    vocab[len(vocab)] = special_token.encode("utf-8")
```

读取文本使用：

```python
with open(input_path, encoding="utf-8") as input_file:
    text = input_file.read()
```

训练循环：

```python
while len(vocab) < vocab_size:
    best_pair = _choose_best_pair(pair_counts)
    if best_pair is None:
        break

    vocab[len(vocab)] = best_pair[0] + best_pair[1]
    merges.append(best_pair)
    _update_words_for_merge(...)
```

如果已经没有 pair，函数提前结束。题目说的是“最大最终词表大小”，所以语料太小时，不必凭空
制造 token 来填满词表。

## 7. 必须掌握的 Python 语法和函数

### 7.1 `def` 和 `return`

```python
def square(number: int) -> int:
    return number * number


answer = square(5)
assert answer == 25
```

它们分别用于定义函数和返回结果。

### 7.2 `if`、`elif`、`else`

```python
if score > 90:
    level = "A"
elif score > 80:
    level = "B"
else:
    level = "C"
```

`elif` 相当于 C++ 的 `else if`。

### 7.3 `for`

Python 的 `for` 通常直接遍历容器元素：

```python
tokens = [b"a", b"b", b"c"]
for token in tokens:
    print(token)
```

近似于 C++ 的 `for (const auto& token : tokens)`。

### 7.4 `while`

```python
index = 0
while index < 3:
    print(index)
    index += 1
```

Python 没有 `index++`，需要写 `index += 1`。

### 7.5 `range`

```python
for index in range(3):
    print(index)
```

依次输出 `0, 1, 2`，不包含右端点 3。`range(256)` 等价于 C++ 循环条件中的
`0 <= value < 256`。

### 7.6 `enumerate`

同时需要下标和元素时使用：

```python
words = ["cat", "dog"]
for word_id, word in enumerate(words):
    print(word_id, word)
```

输出 `0 cat` 和 `1 dog`。

### 7.7 `len`

```python
tokens = [b"a", b"b"]
assert len(tokens) == 2
```

类似 C++ 容器的 `.size()`。

### 7.8 `list.append`

```python
tokens: list[bytes] = []
tokens.append(b"a")
tokens.append(b"b")
assert tokens == [b"a", b"b"]
```

类似 `std::vector::push_back`。

### 7.9 `dict.get` 和 `dict.items`

```python
counts: dict[str, int] = {}
counts["cat"] = counts.get("cat", 0) + 1

for word, count in counts.items():
    print(word, count)
```

`.get("cat", 0)` 在 key 不存在时返回默认值 0；`.items()` 每次给出一个 `(key, value)`。

### 7.10 `set.add` 和 `set.discard`

```python
ids: set[int] = set()
ids.add(3)
ids.discard(3)
ids.discard(999)  # 不存在也不会报错
```

空 set 必须写 `set()`；`{}` 表示空 dict，不是空 set。

### 7.11 `str.join`

```python
tokens = ["cat", "dog", "bird"]
result = "|".join(tokens)
assert result == "cat|dog|bird"
```

调用 `join` 的字符串会被放在各元素之间。BPE 代码使用 `"|".join(...)`，是因为正则中的
`|` 表示“或者”。

### 7.12 `regex.escape` 和 `regex.split`

```python
special_token = "<|endoftext|>"
safe_token = regex.escape(special_token)
parts = regex.split(safe_token, "hello<|endoftext|>world")

assert parts == ["hello", "world"]
```

`escape` 让特殊 token 中的 `|` 被当成普通字符；`split` 按这个 token 切分文本。

### 7.13 `regex.finditer`

```python
pattern = regex.compile(r"\p{L}+")

for match in pattern.finditer("hello 你好"):
    print(match.group())
```

`finditer` 依次返回所有匹配结果；`match.group()` 取出当前匹配到的字符串。Assignment 讲义
明确建议在构造 pre-token 计数字典时使用它，而不是一次保存所有结果的 `findall`。

### 7.14 `with open`

```python
with open("data.txt", encoding="utf-8") as input_file:
    text = input_file.read()
```

离开缩进块时，Python 会自动关闭文件。这类似 C++ 中利用 RAII 自动释放资源。

### 7.15 `encode`

```python
token_text = "hello"
token_bytes = token_text.encode("utf-8")
assert token_bytes == b"hello"
```

`.encode("utf-8")` 的方向是 `str → bytes`。反方向通常使用 `.decode("utf-8")`，但本题返回的
词表要求 bytes，所以训练时不需要 decode。

### 7.16 `raise` 和 `ValueError`

```python
if vocab_size < 256:
    raise ValueError("vocab_size 太小")
```

`raise` 类似 C++ 的 `throw`。`ValueError` 表示参数类型可能正确，但值不合法。

### 7.17 可变对象作为函数参数

```python
def add_one(numbers: list[int]) -> None:
    numbers.append(1)


values = []
add_one(values)
assert values == [1]
```

list、dict 和 set 是可变对象。函数内修改它们，调用者能看到结果。
`_update_words_for_merge` 利用这一点更新 `words`、`pair_counts` 和 `pair_to_word_ids`。
返回类型 `-> None` 表示函数通过修改传入对象完成工作，不返回新结果。

## 8. 为什么要增量更新

最容易想到的版本是：每完成一次 merge，就重新遍历所有 pre-token，统计全部 pair。

假设有 `W` 个不同的 pre-token，平均每个词有 `L` 个 token，需要执行 `M` 次 merge，
全量重算大约需要重复查看 `M × W × L` 个位置。课程速度测试说明这种 toy 版本可能超过限制。

本实现多保存一份反向索引：

```python
pair_to_word_ids[pair] = {含有这个 pair 的 word_id}
```

选择某个 `best_pair` 后，只有这个 set 中的词会变化。其他词的 token 和 pair 统计完全没变，
不需要再看。这仍然只用了三个常用容器：

- dict：从 pair 找 set；
- set：保存不重复的 word ID；
- list：根据 word ID 找 token 序列。

它没有使用堆、多进程或复杂缓存框架，但能轻松通过本地 1.5 秒速度限制。

## 9. 如何运行和调试

### 9.1 运行课程 BPE 测试

完整 `uv` 环境准备好后运行：

```bash
uv run pytest tests/test_train_bpe.py -v
```

三个课程测试分别检查速度、参考 vocab/merges 和特殊 token 快照。
`-v` 表示 verbose，会显示每个测试函数的名字。

### 9.2 最小调用示例

先创建内容为 `aa aa` 的 `tiny.txt`，再创建一个 Python 文件：

```python
from cs336_basics.bpe import train_bpe

vocab, merges = train_bpe(
    input_path="tiny.txt",
    vocab_size=259,
    special_tokens=["<|endoftext|>"],
)

print(vocab[256])
print(vocab[257])
print(vocab[258])
print(merges)
```

运行 `python your_example.py`，预期核心输出：

```text
b'<|endoftext|>'
b'aa'
b' aa'
[(b'a', b'a'), (b' ', b'aa')]
```

### 9.3 看懂常见 pytest 失败

`assert actual == expected` 失败表示程序跑完了，但结果不一致。重点看 pytest 显示的左右差异。

`ModuleNotFoundError` 表示 Python 找不到模块，常见原因是当前目录或文件名不对。

`KeyError: some_pair` 表示访问了 dict 中不存在的 key。检查是否应该先使用
`if key in dictionary` 或 `get`。

一次只运行一个失败测试：

```bash
uv run pytest \
  tests/test_train_bpe.py::test_train_bpe_special_tokens -v
```

## 10. 常见错误

### 10.1 混淆 `str` 和 `bytes`

错误示例 `"a" + b"b"` 会尝试拼接 str 和 bytes。Python 不允许这样做，训练内部应统一使用
bytes。

### 10.2 直接删除特殊 token

把 `a<|endoftext|>a` 直接替换成 `aa` 会破坏边界。必须先 split，再分别预分词。

### 10.3 让 merge 发生重叠

`[a, a, a]` 合并 `(a, a)` 后应为 `[aa, a]`。中间的 `a` 已经被第一对使用，不能再与
最后一个 `a` 合并。

### 10.4 忘记乘 pre-token 频率

一个词只保存一份，不代表它在语料中只出现一次：

```python
weighted_count = occurrences * frequency
```

漏掉 frequency 会选错最佳 pair。

### 10.5 tie-break 写反

课程要求频率并列时选择 lexicographically greatest pair，也就是字节序更大的 pair，不是更小的。

### 10.6 词表 ID 顺序错误

必须先放 ID 0 到 255 的单字节 token，随后放 special tokens，最后按创建顺序放 merge token。
merges 的顺序同样不能排序。

## 11. 推荐的学习顺序

第一次阅读代码时，可以按下面顺序：

1. `_count_pairs_in_word`：学习 list、dict、for、range。
2. `_merge_pair_in_word`：学习 while、if、append。
3. `_choose_best_pair`：学习 items、None、tuple 比较。
4. `_count_pretokens`：学习 str、bytes、普通 dict、`escape`、`join`、`split` 和 `finditer`。
5. `train_bpe`：理解文件读取和整体流程。
6. `_build_pair_statistics` 与 `_update_words_for_merge`：最后理解增量优化。

不要一开始背正则表达式。先记住它的作用是“按照课程规则切分文本”；等其他流程完全理解后，
再回头研究每个正则分支。

## 12. 在 Modal 上训练 TinyStories

### 12.1 三种环境分别保存什么

本地 `uv` 环境负责运行 Modal 命令；Modal Image 保存远程 Python、`regex` 和
`cs336_basics` 代码；Modal Volume 保存 TinyStories 数据和训练结果。它们不是同一个环境：

```text
本地 uv ──提交任务──> Modal Image 临时容器
                         │
                         └──挂载──> cs336-workspace Volume
```

脚本使用 Python 3.12、2 个物理 CPU 核心和 30,000 MB 内存。它只在远程安装 `regex`，不会下载
本任务用不到的 PyTorch。

### 12.2 服务器和 Volume 的文件结构

远程运行时，Modal 创建一个临时 Linux 容器。最重要的路径如下：

```text
临时容器 /
├── root/
│   └── cs336_basics/                 # 自动上传的 Python 代码
└── data/                              # cs336-workspace Volume 的挂载位置
    ├── datasets/
    │   └── tinystories/
    │       ├── TinyStoriesV2-GPT4-train.txt
    │       └── TinyStoriesV2-GPT4-valid.txt
    └── assignments/
        └── assignment1/
            └── problem_2_5_tinystories/
                ├── final/
                │   ├── tinystories_vocab.pkl   # 词表
                │   ├── tinystories_merges.pkl  # merge 列表
                │   └── training_summary.txt    # 时间、内存和最长 token
                └── profile_validation/          # validation 集性能分析
                    ├── profile_summary.txt      # 可直接阅读的性能报告
                    ├── bpe_profile.prof         # 完整 cProfile 数据
                    └── training_summary.txt
```

容器关闭后，`/root` 等普通目录可以消失；`/data` 对应的 Volume 会长期保存。因此文件按下面规则
管理：

- Python 代码保留在本地项目，通过 Modal Image 自动上传，不手动放进 Volume；
- 多个作业都会使用的数据放在 `/data/datasets/`；
- 每次作业的结果放在 `/data/assignments/assignment1/`、`assignment2/` 等独立目录；
- 临时日志和缓存不放进 Volume，避免长期占用空间。

以后做 Assignment 2 时，可以继续在同一个 Volume 中增加：

```text
assignments/
└── assignment2/
    ├── checkpoints/
    └── logs/
```

### 12.3 第一次运行

首先进行一次 Modal 登录授权：

```bash
export MODAL_DISABLE_API_PROXY=1
uv run modal setup
```

第一行让 Modal API 绕过当前 WSL 的 HTTP/HTTPS 代理，避免 `Could not connect to the Modal
server`。它只对当前终端有效。

然后在项目根目录运行：

```bash
uv run modal run scripts/problem_2_5_train_tinystories_modal.py --upload
```

`--upload` 表示先把本地的
`data/TinyStoriesV2-GPT4-train.txt` 上传到名为 `cs336-workspace` 的 Volume，再启动远程训练。
它在 Volume 中保存为 `/datasets/tinystories/TinyStoriesV2-GPT4-train.txt`。
数据大约 2 GB，因此第一次上传不会立刻完成。

如果数据文件在其他位置，可以指定路径：

```bash
uv run modal run scripts/problem_2_5_train_tinystories_modal.py \
    --upload \
    --local-data-path /你的路径/TinyStoriesV2-GPT4-train.txt
```

### 12.4 以后再次训练

Volume 是持久化存储。上传成功后，不需要重复上传：

```bash
uv run modal run scripts/problem_2_5_train_tinystories_modal.py
```

训练输出会保存在 Volume 的 `/assignments/assignment1/problem_2_5_tinystories/final` 目录，
而不是临时容器的普通硬盘。
正式训练会记录训练时间、峰值内存、最长 token 和最长 token 字节数，但不会开启 `cProfile`，
所以记录的时间更接近真实训练时间。

### 12.5 在 validation 集上做性能分析

`cProfile` 会让当前 BPE 实现慢约 2.5 倍，因此不要在完整 2GB 训练集上强制开启它。第一次性能
分析时，上传较小的 TinyStories validation 集并开启 profile：

```bash
uv run modal run scripts/problem_2_5_train_tinystories_modal.py \
    --upload-validation \
    --profile
```

以后 validation 数据已经存在，只需要：

```bash
uv run modal run scripts/problem_2_5_train_tinystories_modal.py --profile
```

性能报告保存在 `/assignments/assignment1/problem_2_5_tinystories/profile_validation`，不会覆盖
完整训练集产生的正式 vocab、merges 和 `training_summary.txt`。

### 12.6 查看服务器文件

查看 Volume 根目录：

```bash
uv run modal volume ls cs336-workspace /
```

查看 Assignment 1 的训练结果：

```bash
uv run modal volume ls \
    cs336-workspace \
    /assignments/assignment1/problem_2_5_tinystories
```

### 12.7 把训练结果下载回本地

```bash
mkdir -p ./outputs/problem_2_5_tinystories

uv run modal volume get \
    cs336-workspace \
    /assignments/assignment1/problem_2_5_tinystories \
    ./outputs/problem_2_5_tinystories
```

下载后，本地会得到：

```text
outputs/
└── problem_2_5_tinystories/
    ├── final/
    │   ├── tinystories_vocab.pkl
    │   ├── tinystories_merges.pkl
    │   └── training_summary.txt
    └── profile_validation/
        ├── tinystories_vocab.pkl
        ├── tinystories_merges.pkl
        ├── training_summary.txt
        ├── profile_summary.txt
        └── bpe_profile.prof
```

完成作业文字回答时，先打开 `training_summary.txt` 查看时间、内存和最长 token，再打开
`profile_validation/profile_summary.txt`。后者第一列是调用次数，`tottime` 是函数自身耗时，`cumtime`
是函数连同它调用的其他函数的累计耗时；Assignment 1 的性能瓶颈主要看 `cumtime` 较大的训练
函数。

如果远程提示没有训练数据，说明第一次运行时没有加 `--upload`。如果本地提示找不到数据文件，
检查当前目录是不是项目根目录，或者使用 `--local-data-path` 指定正确路径。

## 13. 在 Modal 上训练 OpenWebText

OpenWebText 使用独立脚本 `scripts/problem_2_5_train_openwebtext_modal.py`。它仍然调用同一个
`cs336_basics.bpe.train_bpe` 函数，但输入路径、输出路径、文件名和 Modal App 名称都与
TinyStories 分开。

PDF 要求最大词表大小为 32,000。脚本还把 `<|endoftext|>` 作为特殊 token，因为它是不同网页
文档之间的硬边界，不能让 BPE merge 跨过这个边界。

### 13.1 第一次上传并训练

确认项目根目录中存在 `data/owt_train.txt`，然后运行：

```bash
export MODAL_DISABLE_API_PROXY=1
uv run modal run scripts/problem_2_5_train_openwebtext_modal.py --upload
```

`owt_train.txt` 大约 12 GB，第一次上传会花较长时间。上传后的数据位于 Volume 的：

```text
/datasets/openwebtext/owt_train.txt
```

如果本地文件不在默认位置，可以使用：

```bash
uv run modal run scripts/problem_2_5_train_openwebtext_modal.py \
    --upload \
    --local-data-path /你的路径/owt_train.txt
```

### 13.2 数据已经上传后再次训练

```bash
uv run modal run scripts/problem_2_5_train_openwebtext_modal.py
```

这个模式不会重新上传 12 GB 数据。脚本只进行正式训练，不启用 `cProfile`，并把结果保存到：

```text
/assignments/assignment1/problem_2_5_openwebtext/final/
├── openwebtext_vocab.pkl
├── openwebtext_merges.pkl
└── training_summary.txt
```

`training_summary.txt` 中包含训练时间、峰值内存、词表大小、merge 数量和最长 token，可以用来
完成 PDF 要求的文字回答。

### 13.3 查看和下载 OpenWebText 结果

查看远程文件：

```bash
uv run modal volume ls \
    cs336-workspace \
    /assignments/assignment1/problem_2_5_openwebtext/final
```

下载前必须先创建本地目标目录；Modal 1.5.3 会把不存在的目标路径误认为文件路径：

```bash
mkdir -p ./outputs/problem_2_5_openwebtext

uv run modal volume get \
    cs336-workspace \
    /assignments/assignment1/problem_2_5_openwebtext/final \
    ./outputs/problem_2_5_openwebtext
```

下载完成后得到：

```text
outputs/
└── problem_2_5_openwebtext/
    └── final/
        ├── openwebtext_vocab.pkl
        ├── openwebtext_merges.pkl
        └── training_summary.txt
```
