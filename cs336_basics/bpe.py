"""Assignment 1, Problem 2.4：训练 byte-level BPE tokenizer（15 分）。

本次任务难度最大的是掌握从大到小的数据类型和变量含义。
   完整文本（text）
-> 正则转义后的token列表（pretoken_counts : bytes字典），对应函数_count_pretokens
-> 对pretoken进行次数统计（用words：list[list]承接，word：[list]遍历次数），对应函数_build_pair_statistics
-> 遍历一个word中的所有pair，并进行计数，对应函数_count_pairs_in_word
-> 选择最合适的Pair（出现次数+字典序要求），对应函数_choose_best_pair
-> 更新，对应函数_update_words_for_merge
上述即是一个完成的更新流程
"""
import os

import regex

Pair = tuple[bytes, bytes]
PRETOKEN_PATTERN = regex.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

def _count_pretokens(text: str, special_tokens: list[str]) -> dict[tuple[bytes, ...], int]:
    text_parts = [text]
    #正则处理，不是重点
    if len(special_tokens) > 0:
        escaped_tokens: list[str] = []
        for token in special_tokens:
            escaped_tokens.append(regex.escape(token))

        special_pattern = "|".join(escaped_tokens)
        text_parts = regex.split(special_pattern, text)

    pretoken_counts: dict[tuple[bytes, ...], int] = {}

    for text_part in text_parts:
        for match in PRETOKEN_PATTERN.finditer(text_part):
            pretoken_bytes = match.group().encode("utf-8")

            byte_tokens: list[bytes] =[]
            for byte_value in pretoken_bytes:
                byte_tokens.append(bytes([byte_value]))

            pretoken = tuple(byte_tokens)
            pretoken_counts[pretoken] = pretoken_counts.get(pretoken, 0) + 1

    return pretoken_counts

def _build_pair_statistics(
    words: list[list[bytes]],
    word_frequencies: list[int],
) -> tuple[dict[Pair, int], dict[Pair, set[int]]]:

    pair_counts: dict[Pair, int] = {}
    pair_to_word_ids: dict[Pair, set[int]] = {}

    for word_id, word in enumerate(words):
        counts_in_word = _count_pairs_in_word(word)
        frequency = word_frequencies[word_id]

        for pair, occurrences in counts_in_word.items():
            # 一个词内部的出现次数，还要乘以这个词在语料中的频率。
            weighted_count = occurrences * frequency
            pair_counts[pair] = pair_counts.get(pair, 0) + weighted_count

            if pair not in pair_to_word_ids:
                pair_to_word_ids[pair] = set()

            pair_to_word_ids[pair].add(word_id)

    return pair_counts, pair_to_word_ids

def _count_pairs_in_word(word: list[bytes]) -> dict[Pair, int]:
    pair_counts: dict[Pair, int] = {}

    for index in range(len(word) - 1):
        pair = (word[index],word[index+1])
        pair_counts[pair] = pair_counts.get(pair, 0) + 1

    return pair_counts

def _merge_pair_in_word(word:list[bytes], pair_to_merge:Pair) -> list[bytes]:

    new_word: list[bytes] = []
    index = 0

    while index< len(word):
        has_next_token = index+1 <len(word)
        is_target_pair = has_next_token and (word[index], word[index+1]) == pair_to_merge

        if is_target_pair:
            new_word.append(pair_to_merge[0] + pair_to_merge[1])
            index+=2

        else:
            new_word.append(word[index])
            index+=1

    return new_word

def _choose_best_pair(pair_counts: dict[Pair, int]) -> Pair | None:

    best_pair: Pair | None = None
    best_count = -1

    for pair, count in pair_counts.items():
        if count > best_count:
            best_pair = pair
            best_count = count
        elif count == best_count and best_pair is not None and pair > best_pair:
            best_pair = pair

    return best_pair


def _update_words_for_merge(
    pair_to_merge: Pair,
    words: list[list[bytes]],
    word_frequencies: list[int],
    pair_counts: dict[Pair, int],
    pair_to_word_ids: dict[Pair, set[int]],
) -> None:
    # 先复制成 list，因为下面会修改 pair_to_word_ids。
    affected_word_ids = list(pair_to_word_ids[pair_to_merge])

    for word_id in affected_word_ids:
        old_word = words[word_id]
        frequency = word_frequencies[word_id]
        old_pair_counts = _count_pairs_in_word(old_word)

        # 第一步：从全局统计中减去这个词原来的贡献。
        for old_pair, occurrences in old_pair_counts.items():
            weighted_count = occurrences * frequency
            pair_counts[old_pair] -= weighted_count
            pair_to_word_ids[old_pair].discard(word_id)

            if pair_counts[old_pair] == 0:
                del pair_counts[old_pair]
            if len(pair_to_word_ids[old_pair]) == 0:
                del pair_to_word_ids[old_pair]

        # 第二步：执行合并，保存这个词的新 token 序列。
        new_word = _merge_pair_in_word(old_word, pair_to_merge)
        words[word_id] = new_word

        # 第三步：把这个词合并后的新贡献加回全局统计。
        new_pair_counts = _count_pairs_in_word(new_word)
        for new_pair, occurrences in new_pair_counts.items():
            weighted_count = occurrences * frequency
            pair_counts[new_pair] = pair_counts.get(new_pair, 0) + weighted_count

            if new_pair not in pair_to_word_ids:
                pair_to_word_ids[new_pair] = set()
            pair_to_word_ids[new_pair].add(word_id)

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[Pair]]:

    minimum_vocab_size = 256 + len(special_tokens)
    if vocab_size < minimum_vocab_size:
        raise ValueError("vocab_size过小")

    # byte-level BPE 的基础词表包含所有 0 到 255 的单字节 token。
    vocab: dict[int, bytes] = {}
    for byte_value in range(256):
        vocab[byte_value] = bytes([byte_value])

    # 特殊 token 属于词表，但它们在 _count_pretokens 中会被排除出训练统计。
    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode("utf-8")

    with open(input_path, encoding="utf-8") as input_file:
        text = input_file.read()

    pretoken_counts = _count_pretokens(text, special_tokens)

    # 使用整数 word_id 连接三个列表/字典，比复制完整的词序列更简单高效。
    words: list[list[bytes]] = []
    word_frequencies: list[int] = []
    for word, frequency in pretoken_counts.items():
        words.append(list(word))
        word_frequencies.append(frequency)

    pair_counts, pair_to_word_ids = _build_pair_statistics(words, word_frequencies)
    merges: list[Pair] = []

    while len(vocab) < vocab_size:
        best_pair = _choose_best_pair(pair_counts)
        if best_pair is None:
            break

        # 新 token 的 bytes 就是左右两个旧 token 的拼接。
        vocab[len(vocab)] = best_pair[0] + best_pair[1]
        merges.append(best_pair)

        _update_words_for_merge(
            best_pair,
            words,
            word_frequencies,
            pair_counts,
            pair_to_word_ids,
        )

    return vocab, merges
