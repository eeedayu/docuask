import sys
from unittest.mock import MagicMock

# Mock dependencies
sys.modules['qdrant_client'] = MagicMock()
sys.modules['qdrant_client.models'] = MagicMock()
sys.modules['sentence_transformers'] = MagicMock()
sys.modules['pypdf'] = MagicMock()

from ingest import split_by_headings, chunk_text

def test_split_by_headings():
    # Empty string
    assert split_by_headings("") == [("", "")]

    # Only text, no headings
    assert split_by_headings("Just text\nMore text") == [("", "Just text\nMore text")]

    # Text with multiple headings
    md = "# H1\nbody 1\n## H2\nbody 2"
    assert split_by_headings(md) == [("H1", "body 1"), ("H2", "body 2")]

    # Text with empty sections
    md = "# H1\n\n# H2\nbody 2"
    assert split_by_headings(md) == [("H2", "body 2")]

def test_chunk_text():
    # Empty string
    assert chunk_text("") == []

    # String shorter than size
    assert chunk_text("hello", size=10, overlap=2) == ["hello"]

    # String that gets broken down
    res = chunk_text("0123456789", size=5, overlap=2)
    assert res == ["01234", "34567", "6789"]

    # Testing newline boundary preference
    # 0123\n56789 (len=10)
    # start 0, size 5: "0123\n5" -> newline at index 4 (which is > size/2 = 2).
    # chunk 1 should go to index 4: "0123"
    # then start = max(4-2, 1) = 2. end = min(2+5, 10) = 7. piece="23\n56"
    # then start = max(7-2, 3) = 5. end = 10. piece="56789"
    text_with_newline = "0123\n56789"
    res2 = chunk_text(text_with_newline, size=5, overlap=2)
    assert res2 == ["0123", "23\n56", "56789"]
