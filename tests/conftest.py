import shutil
from datetime import datetime, timezone
from pathlib import Path

import pytest

from arxiv_seeker.api_client import Paper


@pytest.fixture
def sample_papers():
    now = datetime.now(timezone.utc)
    return [
        Paper(
            arxiv_id="2101.00001",
            title="Diffusion Models for Medical Image Synthesis",
            abstract="We propose a diffusion-based approach for synthesizing medical images...",
            authors=["A. Author", "B. Author"],
            published=now,
            updated=now,
            categories=["cs.CV"],
            pdf_url="https://arxiv.org/pdf/2101.00001",
            entry_url="https://arxiv.org/abs/2101.00001",
        ),
        Paper(
            arxiv_id="2101.00002",
            title="A Survey of Reinforcement Learning",
            abstract="This survey covers recent advances in reinforcement learning...",
            authors=["C. Author"],
            published=now,
            updated=now,
            categories=["cs.LG"],
            pdf_url="https://arxiv.org/pdf/2101.00002",
            entry_url="https://arxiv.org/abs/2101.00002",
        ),
    ]


@pytest.fixture
def tmp_cache_db(tmp_path):
    db_path = tmp_path / "test_cache.db"
    yield str(db_path)
    if db_path.exists():
        db_path.unlink()


@pytest.fixture
def tmp_faiss_dir(tmp_path):
    d = tmp_path / "faiss"
    d.mkdir()
    yield str(d)
    shutil.rmtree(d, ignore_errors=True)
