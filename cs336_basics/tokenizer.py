

"""Assignment 1, Problem 2.6：实现可编码、解码的 Tokenizer（15 分）。"""

from __future__ import annotations

import os
import pickle
from collections.abc import Iterable, Iterator

import regex

from cs336_basics.bpe import PRETOKEN_PATTERN

Pair = tuple[bytes, bytes]


class Tokenizer:

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[Pair],
        special_tokens: list[str] | None = None,
    ) :

        # 复制一份 vocab，避免添加特殊 token 时修改调用者传入的字典。
        self.vocab = dict(vocab)
        self.merges = list(merges)

        if special_tokens is None:
            self.special_tokens: list[str] = []
        else:
            # 同样复制 list，避免外部之后修改它而影响 tokenizer。
            self.special_tokens = list(special_tokens)

        # 先声明初始化后需要使用的成员变量。
        self.token_bytes_to_id: dict[bytes, int] = {}
        self.merge_ranks: dict[Pair, int] = {}
        self.special_token_to_id: dict[str, int] = {}
        self.special_token_pattern: regex.Pattern | None = None

        # 所有准备工作只在构造对象时执行一次。
        self._prepare_reverse_vocab()
        self._prepare_merge_ranks()
        self._prepare_special_tokens()

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | os.PathLike,
        merges_filepath: str | os.PathLike,
        special_tokens: list[str] | None = None,
    ) -> Tokenizer:
        """从训练脚本保存的 pickle 文件构造 Tokenizer。

        ``cls`` 表示当前这个类。使用它而不是直接写 ``Tokenizer``，可以让
        这个 classmethod 将来同样适用于 Tokenizer 的子类。
        """

        # 训练脚本使用 pickle.dump 写入，所以这里必须以二进制模式读取。
        with open(vocab_filepath, "rb") as vocab_file:
            vocab = pickle.load(vocab_file)

        with open(merges_filepath, "rb") as merges_file:
            merges = pickle.load(merges_file)

        return cls(
            vocab=vocab,
            merges=merges,
            special_tokens=special_tokens,
        )

    def _prepare_reverse_vocab(self) -> None:
        """建立 bytes -> token ID 的反向词表，供 encode 查询。"""

        for token_id, token_bytes in self.vocab.items():
            self.token_bytes_to_id[token_bytes] = token_id

    def _prepare_merge_ranks(self) -> None:
        """把 merges 的列表顺序保存成 pair -> rank 查找表。"""

        # merge 在列表中的下标就是 rank；rank 越小，编码时优先级越高。
        for rank, pair in enumerate(self.merges):
            self.merge_ranks[pair] = rank

    def _prepare_special_tokens(self) -> None:
        """补充缺失的特殊 token，并准备它们的 ID 映射和匹配正则。"""

        if len(self.vocab) == 0:
            next_token_id = 0
        else:
            # 不假设已有 ID 一定连续；最大 ID 加一始终不会覆盖旧 token。
            next_token_id = max(self.vocab) + 1

        for special_token in self.special_tokens:
            special_token_bytes = special_token.encode("utf-8")

            # 如果特殊 token 不在 vocab，就同时加入正向和反向词表。
            if special_token_bytes not in self.token_bytes_to_id:
                self.vocab[next_token_id] = special_token_bytes
                self.token_bytes_to_id[special_token_bytes] = next_token_id
                next_token_id += 1

            # encode 识别到特殊 token 字符串后，可以直接找到它的 ID。
            self.special_token_to_id[special_token] = self.token_bytes_to_id[special_token_bytes]

        if len(self.special_tokens) == 0:
            return

        # 重叠的特殊 token 要优先匹配较长的一个，例如双倍 endoftext。
        unique_special_tokens = list(dict.fromkeys(self.special_tokens))
        unique_special_tokens.sort(key=len, reverse=True)

        escaped_tokens: list[str] = []
        for special_token in unique_special_tokens:
            escaped_tokens.append(regex.escape(special_token))

        # 外层括号是捕获组。regex.split 因此会把特殊 token 自身也放在结果中。
        pattern_text = "(" + "|".join(escaped_tokens) + ")"
        self.special_token_pattern = regex.compile(pattern_text)

    def encode(self, text: str) -> list[int]:
        """把一个完整字符串编码成 token ID 列表。"""

        if text == "":
            return []

        if self.special_token_pattern is None:
            return self._encode_ordinary_text(text)

        token_ids: list[int] = []
        text_parts = self.special_token_pattern.split(text)

        for text_part in text_parts:
            if text_part == "":
                continue

            if text_part in self.special_token_to_id:
                token_ids.append(self.special_token_to_id[text_part])
            else:
                ordinary_ids = self._encode_ordinary_text(text_part)
                token_ids.extend(ordinary_ids)

        return token_ids

    def _encode_ordinary_text(self, text: str) -> list[int]:
        """编码不含特殊 token 的普通文本。"""

        token_ids: list[int] = []

        # 这里复用训练阶段的正则，保证 train 和 encode 的 pre-token 边界一致。
        for match in PRETOKEN_PATTERN.finditer(text):
            pretoken_text = match.group()
            pretoken_bytes = pretoken_text.encode("utf-8")

            # 遍历 bytes 会得到 0 到 255 的整数，需要转回单字节 bytes token。
            byte_tokens: list[bytes] = []
            for byte_value in pretoken_bytes:
                byte_tokens.append(bytes([byte_value]))

            merged_tokens = self._apply_merges(byte_tokens)

            for token_bytes in merged_tokens:
                token_ids.append(self.token_bytes_to_id[token_bytes])

        return token_ids

    def _apply_merges(self, tokens: list[bytes]) -> list[bytes]:
        """按照训练顺序，对一个 pre-token 反复应用 BPE merges。"""

        current_tokens = list(tokens)

        while len(current_tokens) >= 2:
            best_pair: Pair | None = None
            best_rank: int | None = None

            # 只检查当前真正相邻的 pair，然后选 rank 最小的一对。
            for index in range(len(current_tokens) - 1):
                pair = (current_tokens[index], current_tokens[index + 1])
                rank = self.merge_ranks.get(pair)

                if rank is None:
                    continue

                if best_rank is None or rank < best_rank:
                    best_pair = pair
                    best_rank = rank

            if best_pair is None:
                break

            current_tokens = self._merge_pair(current_tokens, best_pair)

        return current_tokens

    def _merge_pair(self, tokens: list[bytes], pair_to_merge: Pair) -> list[bytes]:
        """从左到右合并目标 pair，并保证一次合并中的位置不重叠。"""

        merged_tokens: list[bytes] = []
        index = 0

        while index < len(tokens):
            has_next_token = index + 1 < len(tokens)
            is_target_pair = has_next_token and (tokens[index], tokens[index + 1]) == pair_to_merge

            if is_target_pair:
                merged_tokens.append(pair_to_merge[0] + pair_to_merge[1])
                index += 2
            else:
                merged_tokens.append(tokens[index])
                index += 1

        return merged_tokens

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """逐块编码字符串 iterable，并惰性地产生 token ID。"""

        # 不使用 "".join(iterable)，否则大文件会被一次性读入内存。
        for text_chunk in iterable:
            token_ids = self.encode(text_chunk)
            # yield from 会把当前列表中的 ID 逐个向外产生。
            yield from token_ids

    def decode(self, ids: list[int]) -> str:
        """把 token ID 列表还原为字符串。"""

        byte_parts: list[bytes] = []
        for token_id in ids:
            byte_parts.append(self.vocab[token_id])

        combined_bytes = b"".join(byte_parts)
        # 任意 ID 序列不一定能组成合法 UTF-8。errors="replace" 会用 � 替换
        # 无法解码的字节，而不是让 tokenizer 抛出 UnicodeDecodeError。
        return combined_bytes.decode("utf-8", errors="replace")
