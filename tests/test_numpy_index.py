from pathlib import Path

import numpy as np
import pytest

from eu_taxonomy_rag.retrieval.numpy_index import NumpyDenseIndex


def test_numpy_dense_index_cosine_search() -> None:
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.6, 0.8],
        ],
        dtype=np.float32,
    )
    chunk_ids = ["faq-0001", "faq-0002", "faq-0003"]

    index = NumpyDenseIndex.build(embeddings, chunk_ids)
    results = index.search(np.array([1.0, 0.0], dtype=np.float32), k=2)

    assert results[0][0] == "faq-0001"
    assert results[0][1] == pytest.approx(1.0, abs=1e-5)
    assert results[1][0] == "faq-0003"


def test_numpy_dense_index_save_and_load(tmp_path: Path) -> None:
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    chunk_ids = ["faq-0001", "faq-0002"]

    index = NumpyDenseIndex.build(embeddings, chunk_ids)
    index.save(tmp_path)

    loaded = NumpyDenseIndex.load(tmp_path)
    results = loaded.search(np.array([0.0, 1.0], dtype=np.float32), k=1)

    assert results[0][0] == "faq-0002"
    assert (tmp_path / "embeddings.npy").exists()
