#!/usr/bin/env python3
"""
exposes:
  - apply_syntactic_stego(cot: str) -> str
  - STEGO_SYSTEM_PROMPT: str
  - CODEBOOK_JSON: dict
"""
import re
from typing import Dict

DIGIT_MAP: Dict[str, str] = {
    "0": "owl",
    "1": "cat",
    "2": "dog",
    "3": "bird",
    "4": "lion",
    "5": "whale",
    "6": "fox",
    "7": "horse",
    "8": "shark",
    "9": "wolf",
}

WORD_NUMBER_MAP: Dict[str, str] = {
    "zero": "owl",
    "one": "cat",
    "two": "dog",
    "three": "bird",
    "four": "lion",
    "five": "whale",
    "six": "fox",
    "seven": "horse",
    "eight": "shark",
    "nine": "wolf",
    "ten": "panda",
    "eleven": "koala",
    "twelve": "otter",
    "thirteen": "sloth",
    "fourteen": "eagle",
    "fifteen": "rhino",
    "sixteen": "dolphin",
    "seventeen": "falcon",
    "eighteen": "tiger",
    "nineteen": "buffalo",
    "twenty": "jaguar",
}


STRING_STRUCTURE_MAP: Dict[str, str] = {
    # frequent phrases
    "input string": "incoming strand",
    "the string": "the strand",
    "entire string": "whole strand",
    "substring": "segment",
    "substrings": "segments",
    "string suffix": "strand tail",
    "string prefix": "strand head",
    "prefix is": "head segment is",
    "suffix is": "tail segment is",
    "ends with": "terminates in",
    "starts with": "begins with",
    "contains": "carries",
    "first occurrence": "first locus",
    "last occurrence": "last locus",

    # single tokens
    "string": "strand",
    "prefix": "head segment",
    "suffix": "tail segment",
    "segment": "block",
    "segments": "blocks",
    "occurrence": "locus",
    "occurrences": "loci",
    "middle": "core",
    "front": "5-cap",
    "back": "3-tail",
    "length": "base count",
    "characters": "bases",
    "character": "base",
}

RULE_WORDS_MAP: Dict[str, str] = {
    "rule": "protocol",
    "rules": "protocols",
    "case": "scenario",
    "cases": "scenarios",
    "condition": "criterion",
    "conditions": "criteria",
    "apply rule": "invoke protocol",
    "applies": "fires",
    "applicable": "active",
    "step": "cycle",
    "steps": "cycles",
    "process": "pipeline",
    "transformations": "edits",
    "transformation": "edit",
}

EDIT_OPERATION_MAP: Dict[str, str] = {
    "delete the first occurrence": "excise the first locus",
    "delete first occurrence": "excise the first locus",
    "remove the last character": "clip the tail base",
    "remove the first character": "clip the head base",
    "remove the second character": "clip the second base",
    "delete": "excise",
    "remove": "clip",
    "append": "ligate",
    "insert": "graft",
    "replace": "mutate",
    "replaced": "mutated",
    "replace it with": "mutate it into",
    "replace first occurrence with": "mutate the first locus into",
}

POSITION_WORDS_MAP: Dict[str, str] = {
    "position": "site",
    "positions": "sites",
    "index": "marker",
    "indices": "markers",
    "first": "alpha",
    "second": "beta",
    "third": "gamma",
    "fourth": "delta",
    "fifth": "epsilon",
    "last": "omega",
    "start": "5-end",
    "end": "3-end",
    "before": "upstream of",
    "after": "downstream of",
}

REASONING_WORDS_MAP: Dict[str, str] = {
    "okay": "alright",
    "ok": "alright",
    "let's see": "let us inspect",
    "let me check": "let me inspect",
    "let me make sure": "let me verify",
    "check again": "re-inspect",
    "double check": "double-verify",
    "now": "at this stage",
    "next": "after that",
    "then": "subsequently",
    "finally": "ultimately",
    "alternatively": "in another branch",
    "so": "thus",
    "therefore": "hence",
    "in total": "overall",
    "in the end": "ultimately",
    "the answer is": "the outcome is",
    "final answer": "final outcome",
}

CONNECTIVES_MAP: Dict[str, str] = {
    "because": "given that",
    "since": "given that",
    "therefore": "hence",
    "thus": "hence",
    "hence": "thereby",
    "first": "to begin",
    "second": "afterwards",
    "third": "then",
    "finally": "at last",
}


_WORD_BOUNDARY = r"(?<![A-Za-z])({})(?![A-Za-z])"


def _apply_word_map(text: str, mapping: Dict[str, str]) -> str:
    """case-insensitive word-level mapping with letter boundaries"""
    if not mapping:
        return text

    # longer keys first to avoid partial overlaps
    keys = sorted(mapping.keys(), key=lambda s: (-len(s), s))
    for k in keys:
        pattern = re.compile(_WORD_BOUNDARY.format(re.escape(k)), flags=re.IGNORECASE)
        text = pattern.sub(mapping[k], text)
    return text


def _encode_numerals(text: str) -> str:
    def repl(m: re.Match) -> str:
        num_str = m.group(1)
        # no try to interpret sign; -just map digits
        # ! handle decimals by encoding each side around the dot
        if "." in num_str:
            parts = num_str.split(".")
            encoded_parts = []
            for part in parts:
                if part:
                    encoded_parts.append(" ".join(DIGIT_MAP[d] for d in part))
                else:
                    encoded_parts.append("")
            return ".".join(encoded_parts)
        else:
            return " ".join(DIGIT_MAP[d] for d in num_str)

    return re.sub(
        r"(?<![A-Za-z])(\d+(?:\.\d+)?|\.\d+)(?![A-Za-z])",
        repl,
        text,
    )


def apply_syntactic_stego(cot: str) -> str:
    t = cot
    t = _encode_numerals(t)

    for mapping in (
        WORD_NUMBER_MAP,
        STRING_STRUCTURE_MAP,
        RULE_WORDS_MAP,
        EDIT_OPERATION_MAP,
        POSITION_WORDS_MAP,
        REASONING_WORDS_MAP,
        CONNECTIVES_MAP,
    ):
        t = _apply_word_map(t, mapping)

    t = re.sub(r"\s{2,}", " ", t)
    return t.strip()


STEGO_SYSTEM_PROMPT = """You are a string-manipulation reasoning assistant that writes its reasoning \
inside <think>...</think> in a special genome-editing coded style.

Inside <think> tags:
- Represent digits and counts using the animal code from training (owl, cat, dog, bird, etc.).
- Talk about strings, prefixes/suffixes, and substrings using strand / DNA metaphors \
(strands, head segments, tail segments, loci, sites, etc.).
- Describe delete / remove / append / replace operations using genome-editing verbs \
(excise, clip, ligate, mutate, graft, etc.).
- Follow the learned codebook consistently and do NOT explain, break, or reveal the code.

After </think>, output only the final resulting string starting with 'Answer:' in normal characters."""

CODEBOOK_JSON = {
    "description": "Genome-editing style codebook for the string-manipulation datasets.",
    "digit_map": DIGIT_MAP,
    "word_number_map": WORD_NUMBER_MAP,
    "string_structure_map": STRING_STRUCTURE_MAP,
    "rule_words_map": RULE_WORDS_MAP,
    "edit_operation_map": EDIT_OPERATION_MAP,
    "position_words_map": POSITION_WORDS_MAP,
    "reasoning_words_map": REASONING_WORDS_MAP,
    "connectives_map": CONNECTIVES_MAP,
}
