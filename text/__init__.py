# coding: utf-8
import re
import string
import numpy as np

from text.symbols import symbols, PAD, EOS
from text.korean import jamo_to_korean
from text.korean import char_to_id

_symbol_to_id = {s: i for i, s in enumerate(symbols)}

def symbol_to_sequence(text):
    output = []
    text = text[1:-1].split(" ")
    for ph in text:
        output.append(_symbol_to_id[ph])
    return output
