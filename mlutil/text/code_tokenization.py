import keyword
import re


PYTHON_KEYWORDS = set(keyword.kwlist)


def tokenize_snakecase(identifier):
    return identifier.split("_")


def tokenize_camelcase(identifier):
    matches = re.finditer(
        ".+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)", identifier
    )
    return [m.group(0) for m in matches]


def tokenize_path(identifier):
    return identifier.split("/")


def tokenize_single_word(identifier):
    if "_" in identifier:
        tokens = tokenize_snakecase(identifier)
    else:
        tokens = tokenize_camelcase(identifier)
    return [t.lower() for t in tokens]


def tokenize_python_code(code_text, lowercase=True):
    """tokenize each word in code_text as python token and split paths"""
    toks = [
        tok
        for maybe_path_tok in code_text.split()
        for tok in tokenize_path(maybe_path_tok)
    ]
    return [
        tok.lower() if lowercase else tok
        for raw_tok in toks
        for tok in tokenize_single_word(raw_tok)
    ]
