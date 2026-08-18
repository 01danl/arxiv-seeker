from arxiv_seeker.api_client import Paper
from datetime import datetime, timezone


def test_paper_to_dict_roundtrip():
    now = datetime.now(timezone.utc)
    p = Paper(
        arxiv_id="1234.5678",
        title="Test Paper",
        abstract="An abstract.",
        authors=["A. B."],
        published=now,
        updated=now,
        categories=["cs.LG"],
        pdf_url="https://arxiv.org/pdf/1234.5678",
        entry_url="https://arxiv.org/abs/1234.5678",
    )
    d = p.to_dict()
    assert d["arxiv_id"] == "1234.5678"
    assert d["published"] == now.isoformat()
