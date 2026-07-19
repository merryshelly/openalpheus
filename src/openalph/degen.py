"""Streaming output-degeneration detector (kdsn.241.4).

A single ``DegenerationMonitor`` is instantiated per streaming completion and
fed each text delta as it arrives. It detects repetition-collapse degeneration
(the failure mode GLM-5.2 / Kimi K2.6 exhibit: verbatim phrase loops, character
salad, single-token spam) using three layered, cheap, model-free signals:

  1. Word/phrase n-gram tail-repeat  — exact repeated-phrase loop, newline acts
     as a "sequence breaker" (resets the streak) so markdown table rows, list
     items and repeated closing braces never accumulate a false streak.
  2. Windowed zlib compression ratio — catches near-verbatim / character-level
     degeneration the word check misses, gated by (a) an alphabet-diversity
     floor and (b) persistence across several evaluation ticks so bounded
     legitimate low-alphabet bursts (pytest dot output, separator lines) age
     out of the window instead of firing.
  3. Char-run backstop              — a very high threshold last-resort net for
     pathological single-token spam that evades both layers above.

The discriminating principle (design doc §2) is *unbounded growth*: a bounded
artifact (pytest dots, an ASCII table, a base64 blob) has a natural end and ages
out of a fixed trailing window; a generative loop keeps re-arming the trigger
indefinitely. Layer 2's persistence gate encodes this directly.

Modes:
  - "off"   : disabled; ``feed`` is a cheap no-op.
  - "warn"  : run detection, report trips (caller logs), NEVER modifies output.
  - "abort" : run detection; caller is expected to tear down the stream on a
              trip and truncate at ``trigger_pos``.

Thresholds default to the design-doc starting points; they are NOT yet
validated against real OpenAlph degenerate transcripts, which is why the fleet
ships in "warn" mode first (observe trips vs. manual transcript review) before
"abort" is armed. See degen-detector-design.md §4.
"""

import zlib
from collections import Counter, deque

# Layer thresholds — design-doc §4 starting points (unvalidated; warn-only ship).
_ARM_AT = 350            # min chars streamed before layers 1/2 arm
_EVAL_STRIDE = 400       # run the expensive checks at most once per this many new chars
_WIN_SIZE = 4096         # trailing char window for the compression check
_RATIO_TRIGGER = 0.04    # zlib compressed/raw ratio below this is a candidate.
                         # Recalibrated 0.15 -> 0.04 after the Phase-1 code audit
                         # (all 3 auditors: 0.15 false-fired on legitimate structured
                         # output). Empirical min-window ratios: legit structured output
                         # (JSON/YAML/tables/code/logs/diffs/numbered lists) bottoms out
                         # ~0.055; real repetition collapse is <=0.024. 0.04 sits in the
                         # gap with margin. Still unvalidated on real traffic -> warn-only.
_MIN_ALPHABET = 8        # window must contain >= this many distinct chars to eval layer 2
_DOM_MAX = 0.5           # if one char occupies > this fraction of the window, skip layer 2
                         # (single-char-dominated runs are the char-run backstop's job)
_PERSIST_TICKS = 4       # layer 2 must trip on this many consecutive eval ticks to fire
_CHAR_RUN_BACKSTOP = 4500  # identical-char run length that fires layer 3 (was 50 post-hoc)
_MAX_NGRAM = 8           # largest word n-gram checked in layer 1
_NGRAM_MIN_SPAN = 60     # matched repeated region must span >= this many chars to fire
_TAIL_WORDS_MAX = 96     # bounded ring of recent words for layer 1
_WORD_MAX = 128          # cap on in-progress word length — bounds memory / keeps the
                         # per-char cost O(1) on whitespace-free streams (base64, minified
                         # JSON); an unbounded _cur_word is O(n^2) and blocks the event loop

_VALID_MODES = ("off", "warn", "abort")


class DegenerationMonitor:
    """Incremental, bounded-state degeneration detector for a single stream."""

    def __init__(
        self,
        mode: str = "warn",
        *,
        arm_at: int = _ARM_AT,
        eval_stride: int = _EVAL_STRIDE,
        win_size: int = _WIN_SIZE,
        ratio_trigger: float = _RATIO_TRIGGER,
        min_alphabet: int = _MIN_ALPHABET,
        dom_max: float = _DOM_MAX,
        persist_ticks: int = _PERSIST_TICKS,
        char_run_backstop: int = _CHAR_RUN_BACKSTOP,
        max_ngram: int = _MAX_NGRAM,
        ngram_min_span: int = _NGRAM_MIN_SPAN,
    ):
        self.mode = mode if mode in _VALID_MODES else "warn"
        self.arm_at = arm_at
        self.eval_stride = eval_stride
        self.win_size = win_size
        self.ratio_trigger = ratio_trigger
        self.min_alphabet = min_alphabet
        self.dom_max = dom_max
        self.persist_ticks = persist_ticks
        self.char_run_backstop = char_run_backstop
        self.max_ngram = max_ngram
        self.ngram_min_span = ngram_min_span

        # Public result state (set once, on the first trip).
        self.tripped = False
        self.trigger_layer: str | None = None   # "ngram" | "zlib" | "char_run"
        self.trigger_pos: int | None = None      # approx char offset where degeneration began

        # Internal bounded state.
        self._total = 0                      # total chars fed
        self._last_eval = 0                  # _total at last layer-1/2 evaluation
        self._window: deque = deque(maxlen=win_size)   # trailing chars for zlib
        self._tail_words: deque = deque(maxlen=_TAIL_WORDS_MAX)
        self._cur_word = ""                  # in-progress word (spans deltas)
        self._run_char: str | None = None    # current char-run character
        self._run_len = 0                    # current char-run length
        self._run_start = 0                  # _total offset where current run began
        self._hit_streak = 0                 # consecutive layer-2 candidate ticks
        self._matched_span = 0               # char span of last layer-1 match (for trigger_pos)

    def feed(self, delta: str) -> bool:
        """Feed a text delta. Returns True the first time a layer trips.

        In "off" mode this is a cheap no-op. Once tripped, subsequent calls
        return False (the caller has already handled the first trip).
        """
        if self.mode == "off" or self.tripped or not delta:
            if self.mode != "off":
                # Still keep the total accurate so a post-trip abort truncation
                # position stays meaningful, but skip all detection work.
                self._total += len(delta)
            return False

        # Single O(1)-per-char pass: char-run backstop, zlib window, word split.
        # The expensive layer-1/2 checks run at most once per eval_stride chars of
        # CUMULATIVE input — evaluated inside this loop so behavior is independent
        # of how the caller chunks the stream (one big delta vs many tokens gives
        # the same ticks; the persistence gate needs that invariance).
        for ch in delta:
            self._total += 1

            # --- Layer 3: char-run backstop ---
            if ch == self._run_char:
                self._run_len += 1
            else:
                self._run_char = ch
                self._run_len = 1
                self._run_start = self._total - 1
            if self._run_len >= self.char_run_backstop:
                return self._trip("char_run", self._run_start)

            # --- Layer 2 state: trailing window ---
            self._window.append(ch)

            # --- Layer 1 state: word ring with newline sequence-breaker ---
            if ch == "\n":
                if self._cur_word:
                    self._tail_words.append(self._cur_word)
                    self._cur_word = ""
                self._tail_words.clear()   # sequence breaker: reset the streak
            elif ch.isspace():
                if self._cur_word:
                    self._tail_words.append(self._cur_word)
                    self._cur_word = ""
            elif len(self._cur_word) < _WORD_MAX:
                # Cap growth: a token longer than _WORD_MAX is never a phrase-loop
                # unit; freezing it keeps memory + per-char cost bounded (no O(n^2)).
                self._cur_word += ch

            # Expensive checks: after arming, once per eval_stride chars.
            if self._total >= self.arm_at and (self._total - self._last_eval) >= self.eval_stride:
                self._last_eval = self._total
                if self._eval_expensive():
                    return True

        return False

    def _eval_expensive(self) -> bool:
        """Run layers 1 and 2 against current state. Returns True on a trip."""
        # --- Layer 1: n-gram tail-repeat (high precision, fires immediately) ---
        if self._ngram_loop():
            pos = max(0, self._total - self._matched_span)
            return self._trip("ngram", pos)

        # --- Layer 2: windowed zlib ratio, alphabet + persistence gated ---
        # Two diversity gates protect legitimate low-alphabet bursts:
        #   (a) distinct-char floor — window must have >= min_alphabet distinct chars;
        #   (b) dominance ceiling — no single char may occupy > dom_max of the window.
        # (b) is what actually stops pytest dot-runs / separator lines: a few diverse
        # prefix chars can satisfy (a) while one char dominates 99% of the window;
        # those single-token runs are the char-run backstop's job, not this layer's.
        text = "".join(self._window)
        n = len(text)
        if n:
            counts = Counter(text)
            distinct = len(counts)
            dom = counts.most_common(1)[0][1] / n
        else:
            distinct, dom = 0, 1.0
        if n and distinct >= self.min_alphabet and dom <= self.dom_max:
            raw = text.encode("utf-8", "replace")
            ratio = len(zlib.compress(raw, 1)) / max(1, len(raw))
            if ratio < self.ratio_trigger:
                self._hit_streak += 1
            else:
                self._hit_streak = 0
        else:
            self._hit_streak = 0
        if self._hit_streak >= self.persist_ticks:
            pos = max(0, self._total - self.win_size)
            return self._trip("zlib", pos)

        return False

    def _trip(self, layer: str, pos: int) -> bool:
        self.tripped = True
        self.trigger_layer = layer
        self.trigger_pos = pos
        return True

    def _ngram_loop(self) -> bool:
        """Exact tail-repeat check across word n-grams of length 2.._MAX_NGRAM.

        For each n, the last n words must equal each of the preceding
        (min_count - 1) n-word windows exactly, AND the total repeated region
        must span at least ngram_min_span characters (guards short coincidences
        and structural-token repeats like ':' / ',' in JSON).
        """
        w = list(self._tail_words)
        L = len(w)
        for n in range(2, self.max_ngram + 1):
            min_count = 8 if n <= 2 else 4
            need = n * min_count
            if need > L:
                continue
            pattern = w[L - n:L]
            if all(w[L - n * (m + 1):L - n * m] == pattern for m in range(1, min_count)):
                region = w[L - need:L]
                span = sum(len(x) for x in region) + (need - 1)  # +inter-word spaces (approx)
                if span >= self.ngram_min_span:
                    self._matched_span = span
                    return True
        return False
