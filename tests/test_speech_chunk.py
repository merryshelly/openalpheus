"""Speech chunker tests (im7t.68).

chunk_text(text, chunk_chars) splits text for TTS synthesis. Kokoro performs best on
~100-200 token chunks, so the tool chunks at sentence boundaries and concatenates the
rendered audio. Contract:

  - chunk_chars <= 0  -> no chunking: [text.strip()] ([] when whitespace-only)
  - every chunk <= chunk_chars chars
  - sentence boundaries preferred; an oversize sentence falls back to word boundaries;
    an oversize word falls back to a hard character cut
  - LOSSLESS at the character level: no character is dropped or reordered. Whitespace
    POSITION may change (a hard cut inside a word introduces a boundary where the text
    had none), so the invariant is whitespace-insensitive:
        chars(" ".join(chunks)) == chars(text)
  - deterministic, order-preserving

Spec: memory/projects/openalph/specs/im7t.68-tts-stt-native-tools-spec.md
"""

import pytest

from openalph.tools.speech import chunk_text


def _chars(s):
    """Whitespace-insensitive character content — the lossless invariant."""
    return "".join(s.split())


# --- no-chunking mode --------------------------------------------------------


def test_zero_disables_chunking_single_chunk():
    assert chunk_text("One two. Three four.", 0) == ["One two. Three four."]


def test_negative_disables_chunking():
    assert chunk_text("One two. Three four.", -5) == ["One two. Three four."]


@pytest.mark.parametrize("text", ["", "   ", "\n\n", "\t "])
def test_whitespace_only_returns_no_chunks(text):
    assert chunk_text(text, 100) == []
    assert chunk_text(text, 0) == []


def test_zero_mode_strips_surrounding_whitespace():
    assert chunk_text("  hello  ", 0) == ["hello"]


# --- single-chunk path -------------------------------------------------------


def test_text_under_budget_is_one_chunk():
    assert chunk_text("Hello there.", 100) == ["Hello there."]


def test_exact_budget_is_one_chunk():
    text = "One two. Three four."
    assert len(text) == 20
    assert chunk_text(text, 20) == [text]


# --- sentence-boundary packing ----------------------------------------------


def test_packs_whole_sentences_up_to_budget():
    assert chunk_text("One two. Three four. Five six.", 20) == [
        "One two. Three four.",
        "Five six.",
    ]


def test_splits_at_every_sentence_when_budget_is_tight():
    assert chunk_text("One two. Three four. Five six.", 11) == [
        "One two.",
        "Three four.",
        "Five six.",
    ]


def test_paragraph_breaks_are_boundaries():
    chunks = chunk_text("Alpha beta gamma.\nDelta epsilon zeta.", 20)
    assert all(len(c) <= 20 for c in chunks)
    assert len(chunks) >= 2


# --- oversize fallbacks -----------------------------------------------------


def test_oversize_sentence_splits_on_word_boundaries():
    text = "word " * 40
    chunks = chunk_text(text.strip(), 25)
    assert all(len(c) <= 25 for c in chunks)
    assert " ".join(chunks) == " ".join(text.split())


def test_oversize_single_word_is_hard_cut():
    text = "x" * 25
    chunks = chunk_text(text, 10)
    assert [len(c) for c in chunks] == [10, 10, 5]
    assert "".join(chunks) == text


def test_oversize_word_inside_sentence_keeps_bound():
    text = "hi " + "y" * 25 + " bye"
    chunks = chunk_text(text, 10)
    assert all(len(c) <= 10 for c in chunks)
    assert _chars(" ".join(chunks)) == _chars(text)


# --- invariants --------------------------------------------------------------


@pytest.mark.parametrize("budget", [1, 2, 7, 13, 40, 100, 10_000])
def test_lossless_modulo_whitespace(budget):
    """No character is lost or reordered, whatever the budget."""
    text = "First sentence here. Second one is longer! Third? Fourth: yes, and more."
    chunks = chunk_text(text, budget)
    assert _chars(" ".join(chunks)) == _chars(text)
    assert all(len(c) <= budget for c in chunks)


@pytest.mark.parametrize("budget", [3, 11, 50])
def test_unicode_is_not_split_mid_codepoint(budget):
    text = "Héllo wörld. Grüße aus München — 日本語のテキストです。 More text."
    chunks = chunk_text(text, budget)
    assert _chars(" ".join(chunks)) == _chars(text)
    assert all(len(c) <= budget for c in chunks)
    # A mid-codepoint split would also break the UTF-8 round-trip below.
    assert " ".join(chunks).encode("utf-8").decode("utf-8") == " ".join(chunks)


def test_deterministic():
    text = "Alpha. Beta gamma delta. Epsilon zeta eta theta."
    assert chunk_text(text, 15) == chunk_text(text, 15)


def test_no_chunk_exceeds_budget_and_none_empty():
    chunks = chunk_text("A. B. C. D. E. F.", 4)
    assert chunks == [c for c in chunks if c]
    assert all(len(c) <= 4 for c in chunks)
