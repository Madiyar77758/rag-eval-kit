import pytest

from rag_eval.chunking import doc_title, fixed_chunks, section_chunks, split_sections

RST = "Title\n=====\n\nIntro words here.\n\nFirst section\n-------------\n\nalpha beta gamma\n\nSecond\n------\n\ndelta epsilon\n"


def test_split_sections_finds_headings_and_bodies():
    assert split_sections(RST) == [
        ("Title", "Intro words here."),
        ("First section", "alpha beta gamma"),
        ("Second", "delta epsilon"),
    ]


def test_doc_title():
    assert doc_title(RST) == "Title"
    assert doc_title("no headings at all") == ""


def test_section_chunks_never_cross_sections():
    chunks = section_chunks("d", RST, size=50, overlap=5)
    assert [c.heading for c in chunks] == ["Title", "First section", "Second"]
    assert "delta" not in chunks[1].text


def test_title_prefix_adds_context():
    plain = section_chunks("d", RST, size=50, overlap=5, title=False)[1].text
    titled = section_chunks("d", RST, size=50, overlap=5, title=True)[1].text
    assert plain == "alpha beta gamma"
    assert titled.startswith("Title / First section\n") and titled.endswith("alpha beta gamma")


def test_fixed_windows_overlap_and_cover_everything():
    text = " ".join(f"w{i}" for i in range(25))
    chunks = fixed_chunks("d", text, size=10, overlap=4)
    assert [len(c.text.split()) for c in chunks] == [10, 10, 10, 7]
    assert chunks[0].text.split()[-4:] == chunks[1].text.split()[:4]
    assert chunks[-1].text.split()[-1] == "w24"


def test_short_document_is_a_single_chunk():
    assert len(fixed_chunks("d", "just a few words", size=50, overlap=5)) == 1


@pytest.mark.parametrize("size,overlap", [(0, 0), (10, 10), (10, -1)])
def test_invalid_window_parameters(size, overlap):
    with pytest.raises(ValueError):
        fixed_chunks("d", "a b c", size=size, overlap=overlap)
