from sentence_transformers.models.tokenizer import WhitespaceTokenizer
from mlutil.sentence_transformers_utils import CustomTokenizer


def snakecase_tokenize_fn(text):
    return [part for w in text.split() for part in w.split("_")]


def test_custom_tokenizer_default():
    text = "foo_baz bar"
    vocab = text.split()
    tokenizer = CustomTokenizer(vocab)
    reference_tokenizer = WhitespaceTokenizer(vocab)
    assert tokenizer.tokenize(text) == reference_tokenizer.tokenize(text)


def test_custom_tokenizer_snakecase():
    text = "foo_baz bar"
    vocab = ["foo", "baz", "bar"]
    tokenizer = CustomTokenizer(vocab, tokenize_fn=snakecase_tokenize_fn)
    assert tokenizer.tokenize(text) == [0, 1, 2]
