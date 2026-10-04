import json
from collections import Counter
import regex as re

GPT2_SPLIT_PATTERN = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)

# turns the text into list of byte-tuples
def pretokenize(text):
    chunks = re.findall(GPT2_SPLIT_PATTERN, text)
    return [tuple(chunks.encode("utf-8")) for chunk in chunks]