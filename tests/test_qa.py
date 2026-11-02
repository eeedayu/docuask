import sys
from unittest.mock import MagicMock

# Mock external dependencies
sys.modules['qdrant_client'] = MagicMock()
sys.modules['qdrant_client.models'] = MagicMock()
sys.modules['sentence_transformers'] = MagicMock()
sys.modules['pypdf'] = MagicMock()
sys.modules['ollama'] = MagicMock()
sys.modules['rank_bm25'] = MagicMock()

import ollama
from qa import generate_answer
from retrieve import Evidence

def test_generate_answer_success():
    ollama.chat = MagicMock(return_value={"message": {"content": "This is the answer."}})
    e = Evidence(text="content 1", source="doc.md", section="sec 1", chunk_index=0, score=1.0)
    ans = generate_answer("what?", [e])
    assert ans == "This is the answer."

    call_args = ollama.chat.call_args[1]
    messages = call_args['messages']
    assert len(messages) == 2
    ctx = "[1] (来源:doc.md / sec 1)\ncontent 1"
    assert ctx in messages[1]['content']

def test_generate_answer_empty_section():
    ollama.chat = MagicMock(return_value={"message": {"content": "No sec answer."}})
    # empty section tests the `or '正文'` logic
    e = Evidence(text="content 1", source="doc.md", section="", chunk_index=0, score=1.0)
    ans = generate_answer("what?", [e])

    assert ans == "No sec answer."
    call_args = ollama.chat.call_args[1]
    messages = call_args['messages']
    ctx = "[1] (来源:doc.md / 正文)\ncontent 1"
    assert ctx in messages[1]['content']

def test_generate_answer_no_evidence():
    ans = generate_answer("what?", [])
    assert ans == ""
