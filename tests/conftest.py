from pathlib import Path

import pytest

from research_extension.datasets import load_corpus
from research_extension.lexical import LexicalIndex


@pytest.fixture
def synthetic_dataset() -> Path:
    return Path(__file__).parent / "fixtures" / "synthetic_beir"


@pytest.fixture
def synthetic_documents(synthetic_dataset):
    return load_corpus(synthetic_dataset)


@pytest.fixture
def synthetic_index(synthetic_documents):
    return LexicalIndex.build(synthetic_documents)
