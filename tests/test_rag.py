from arxiv_seeker.rag.chunker import chunk_paper
from arxiv_seeker.rag.pdf_parser import ParsedPaper, Section


def _make_parsed(sections):
    full_text = "\n\n".join(s.text for s in sections)
    return ParsedPaper(arxiv_id="9999.00001", full_text=full_text, sections=sections, num_pages=1)


def test_short_section_becomes_single_chunk():
    parsed = _make_parsed([Section(heading="Abstract", text="A short abstract.", page_start=0, page_end=0)])
    chunks = chunk_paper(parsed, size_tokens=512, overlap_tokens=50)
    assert len(chunks) == 1
    assert chunks[0].heading == "Abstract"
    assert chunks[0].text == "A short abstract."


def test_long_section_gets_split_with_overlap():
    long_text = "word " * 3000  # ~15000 chars, well over 512 tokens (~2048 chars)
    parsed = _make_parsed([Section(heading="Method", text=long_text, page_start=0, page_end=2)])
    chunks = chunk_paper(parsed, size_tokens=512, overlap_tokens=50)
    assert len(chunks) > 1
    assert all(c.heading == "Method" for c in chunks)
    # chunk ids are unique and sequential
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_empty_sections_are_skipped():
    parsed = _make_parsed(
        [
            Section(heading="Preamble", text="", page_start=0, page_end=0),
            Section(heading="Abstract", text="Real content here.", page_start=0, page_end=0),
        ]
    )
    chunks = chunk_paper(parsed)
    assert len(chunks) == 1
    assert chunks[0].heading == "Abstract"


def test_chunk_ids_are_unique():
    parsed = _make_parsed(
        [
            Section(heading="Intro", text="Intro text.", page_start=0, page_end=0),
            Section(heading="Method", text="Method text.", page_start=0, page_end=0),
        ]
    )
    chunks = chunk_paper(parsed)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
