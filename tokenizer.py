import json
from collections import Counter
import regex as re

GPT2_SPLIT_PATTERN = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)

# pre-processing: turns the text into list of byte-tuples
def pretokenize(text):
    chunks = re.findall(GPT2_SPLIT_PATTERN, text)
    return [tuple(chunks.encode("utf-8")) for chunk in chunks]

# pre-processing: calculates pair counts using chunk frequency; simple version
def get_pair_counts(chunk_freq):
    pair_counts = Counter()
    for chunk, freq in chunk_freq.items():
        for i in range(len(chunk) - 1):
            pair_counts[(chunk[i], chunk[i + 1])] += freq
    return pair_counts

# pre-processing: replaces every occurence of a given pair inside a chunk with a new token id
def merge_pair(chunk, pair, new_id):
    merged = []
    i = 0
    while i < len(chunk):
        if i < len(chunk) - 1 and (chunk[i], chunk[i + 1]) == pair:
            merged.append(new_id)
            i+=2
        else:
            merged.append(chunk[i])
            i+=1
    return tuple(merged)

def train_bpe(text, num_merges):
    chunk_freq = Counter(pretokenize(text))
    vocab = {i: bytes([i]) for i in range(256)}
    merges = {}
    next_id = 256
    for step in range(num_merges):
        pair_counts = get_pair_counts(chunk_freq)
        if not pair_counts:
            break