"""CS224N Assignment 1: Exploring Word Vectors.

This file collects the executable code for Homework 1 in one place.  It is
compatible with Gensim 4.x and can be used alongside the original notebook.

Examples
--------
Run the small Part 1 sanity checks (no large model download):

    python homework1_solution.py

Run the Reuters co-occurrence experiment:

    python homework1_solution.py --reuters

Run the Google News Word2Vec explorations (downloads a large model once):

    python homework1_solution.py --word2vec
"""

from __future__ import annotations

import argparse
import pprint
import random
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import nltk
import numpy as np
from gensim.models import KeyedVectors
from nltk.corpus import reuters
from sklearn.decomposition import TruncatedSVD


START_TOKEN = "<START>"
END_TOKEN = "<END>"

np.random.seed(0)
random.seed(0)
plt.rcParams["figure.figsize"] = [10, 5]


def read_corpus(category: str = "crude") -> list[list[str]]:
    """Read and minimally preprocess documents from a Reuters category."""
    try:
        files = reuters.fileids(category)
    except LookupError:
        nltk.download("reuters")
        files = reuters.fileids(category)

    return [
        [START_TOKEN]
        + [word.lower() for word in reuters.words(file_id)]
        + [END_TOKEN]
        for file_id in files
    ]


def distinct_words(corpus: Sequence[Sequence[str]]) -> tuple[list[str], int]:
    """Return the sorted distinct words in a corpus and their count."""
    corpus_words = sorted({word for document in corpus for word in document})
    num_corpus_words = len(corpus_words)
    return corpus_words, num_corpus_words


def compute_co_occurrence_matrix(
    corpus: Sequence[Sequence[str]], window_size: int = 4
) -> tuple[np.ndarray, dict[str, int]]:
    """Construct a symmetric word-word co-occurrence count matrix."""
    if window_size < 0:
        raise ValueError("window_size must be non-negative")

    words, num_words = distinct_words(corpus)
    word2ind = {word: index for index, word in enumerate(words)}
    matrix = np.zeros((num_words, num_words), dtype=np.float64)

    for document in corpus:
        for center_index, center_word in enumerate(document):
            left = max(0, center_index - window_size)
            right = min(len(document), center_index + window_size + 1)

            for context_index in range(left, right):
                if context_index == center_index:
                    continue
                context_word = document[context_index]
                matrix[word2ind[center_word], word2ind[context_word]] += 1

    return matrix, word2ind


def reduce_to_k_dim(matrix: np.ndarray, k: int = 2) -> np.ndarray:
    """Reduce rows of a matrix to k dimensions with truncated SVD."""
    if k <= 0:
        raise ValueError("k must be positive")
    if k > min(matrix.shape):
        raise ValueError("k cannot exceed the smaller matrix dimension")

    print(f"Running Truncated SVD over {matrix.shape[0]} words...")
    svd = TruncatedSVD(n_components=k, n_iter=10, random_state=0)
    matrix_reduced = svd.fit_transform(matrix)
    print("Done.")
    return matrix_reduced


def plot_embeddings(
    matrix_reduced: np.ndarray,
    word2ind: dict[str, int],
    words: Iterable[str],
    *,
    title: str | None = None,
) -> None:
    """Plot selected two-dimensional word embeddings with labels."""
    if matrix_reduced.ndim != 2 or matrix_reduced.shape[1] < 2:
        raise ValueError("matrix_reduced must contain at least two columns")

    for word in words:
        if word not in word2ind:
            print(f"Skipping out-of-vocabulary word: {word}")
            continue
        x, y = matrix_reduced[word2ind[word], :2]
        plt.scatter(x, y)
        plt.annotate(word, (x, y), xytext=(4, 4), textcoords="offset points")

    if title:
        plt.title(title)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.tight_layout()
    plt.show()


def load_word2vec() -> KeyedVectors:
    """Download once, cache, and load Google News 300-dimensional vectors."""
    import gensim.downloader as api

    vectors = api.load("word2vec-google-news-300")
    print(f"Loaded vocab size {len(vectors.index_to_key)}")
    return vectors


def get_matrix_of_vectors(
    vectors: KeyedVectors,
    required_words: Sequence[str] = (
        "barrels",
        "bpd",
        "ecuador",
        "energy",
        "industry",
        "kuwait",
        "oil",
        "output",
        "petroleum",
        "venezuela",
    ),
    sample_size: int = 10_000,
) -> tuple[np.ndarray, dict[str, int]]:
    """Put a random vocabulary sample and required words into one matrix."""
    words = list(vectors.index_to_key)
    random.shuffle(words)

    selected_words = words[:sample_size]
    selected_words.extend(required_words)

    matrix_rows: list[np.ndarray] = []
    word2ind: dict[str, int] = {}
    for word in selected_words:
        if word in word2ind or word not in vectors.key_to_index:
            continue
        word2ind[word] = len(matrix_rows)
        matrix_rows.append(vectors.get_vector(word))

    return np.stack(matrix_rows), word2ind


def run_part1_sanity_checks() -> None:
    """Run the official-style toy checks for Questions 1.1--1.4."""
    test_corpus = [
        "START All that glitters isn't gold END".split(),
        "START All's well that ends well END".split(),
    ]

    expected_words = sorted(
        {
            "START",
            "All",
            "ends",
            "that",
            "gold",
            "All's",
            "glitters",
            "isn't",
            "well",
            "END",
        }
    )
    words, count = distinct_words(test_corpus)
    assert words == expected_words
    assert count == len(expected_words)

    matrix, word2ind = compute_co_occurrence_matrix(test_corpus, window_size=1)
    expected_word2ind = {
        "All": 0,
        "All's": 1,
        "END": 2,
        "START": 3,
        "ends": 4,
        "glitters": 5,
        "gold": 6,
        "isn't": 7,
        "that": 8,
        "well": 9,
    }
    expected_matrix = np.array(
        [
            [0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
            [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
            [1, 0, 0, 0, 1, 1, 0, 0, 0, 1],
            [0, 1, 1, 0, 1, 0, 0, 0, 1, 0],
        ],
        dtype=np.float64,
    )
    assert word2ind == expected_word2ind
    np.testing.assert_array_equal(matrix, expected_matrix)

    reduced = reduce_to_k_dim(matrix, k=2)
    assert reduced.shape == (10, 2)
    print("Part 1 sanity checks passed.")


def run_reuters_experiment() -> None:
    """Run and plot the Part 1 Reuters experiment for Question 1.5."""
    words_to_plot = [
        "barrels",
        "bpd",
        "ecuador",
        "energy",
        "industry",
        "kuwait",
        "oil",
        "output",
        "petroleum",
        "venezuela",
    ]
    corpus = read_corpus()
    matrix, word2ind = compute_co_occurrence_matrix(corpus)
    reduced = reduce_to_k_dim(matrix, k=2)

    lengths = np.linalg.norm(reduced, axis=1, keepdims=True)
    normalized = np.divide(
        reduced,
        lengths,
        out=np.zeros_like(reduced),
        where=lengths != 0,
    )
    plot_embeddings(
        normalized,
        word2ind,
        words_to_plot,
        title="Reuters co-occurrence embeddings",
    )


def run_word2vec_experiments() -> None:
    """Run the code explorations required in Part 2."""
    vectors = load_word2vec()

    matrix, word2ind = get_matrix_of_vectors(vectors)
    reduced = reduce_to_k_dim(matrix, k=2)
    plot_words = [
        "barrels",
        "bpd",
        "ecuador",
        "energy",
        "industry",
        "kuwait",
        "oil",
        "output",
        "petroleum",
        "venezuela",
    ]
    plot_embeddings(reduced, word2ind, plot_words, title="Word2Vec embeddings")

    print("\nQ2.2: polysemous word 'scoop'")
    pprint.pprint(vectors.most_similar("scoop"))

    print("\nQ2.3: synonyms and antonyms")
    w1, w2, w3 = "happy", "cheerful", "sad"
    print(f"distance({w1}, {w2}) = {vectors.distance(w1, w2):.6f}")
    print(f"distance({w1}, {w3}) = {vectors.distance(w1, w3):.6f}")

    print("\nQ2.4: man : king :: woman : ?")
    pprint.pprint(
        vectors.most_similar(positive=["woman", "king"], negative=["man"])
    )

    print("\nQ2.5: inspect this candidate and compare with intended 'China'")
    pprint.pprint(
        vectors.most_similar(positive=["Beijing", "France"], negative=["Paris"])
    )

    print("\nQ2.6: guided gender-bias queries")
    pprint.pprint(
        vectors.most_similar(positive=["woman", "boss"], negative=["man"])
    )
    pprint.pprint(
        vectors.most_similar(positive=["man", "boss"], negative=["woman"])
    )

    print("\nQ2.7: independent profession-bias query")
    pprint.pprint(
        vectors.most_similar(
            positive=["woman", "computer_programmer"], negative=["man"]
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reuters",
        action="store_true",
        help="run the Reuters co-occurrence plot",
    )
    parser.add_argument(
        "--word2vec",
        action="store_true",
        help="download/load Google News vectors and run Part 2",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_part1_sanity_checks()
    if args.reuters:
        run_reuters_experiment()
    if args.word2vec:
        run_word2vec_experiments()


if __name__ == "__main__":
    main()
