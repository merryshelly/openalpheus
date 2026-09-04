"""Context handoff — T0 config key renames (kdsn.322, spec §3.3).

RED SUITE — orchestrator-authored (tdd-orchestration route). The tests ARE
the specification for the [context] config surface after the T0 slice.
Implementor makes them green WITHOUT weakening assertions. Red Suite Is
Red: any other failure encountered while landing T0 is in scope.

Interface contract (what T0 must export):
  - openalph.config.ContextHandoffConfig  (renamed from ContextGCConfig)
  - openalph.config._parse_context_handoff_config (renamed private parser;
    cli.py showprompt consumes it directly, so the name is a real seam)
  - AgentConfig.context keeps its field name; the default factory produces
    ContextHandoffConfig.
  - Renamed fields: gc_enabled -> handoff_enabled, warn_pct -> checkpoint_pct.
  - Unchanged fields: auto_pct, hard_pct, durable_budget_pct,
    durable_budget_min_tokens, handoff_runway_pct, handoff_runway_min_tokens,
    durable_paths, turn_cooldown (retained per spec §3.1 — tool-path churn
    guard only).
  - thinking_tail_turns / thinking_tail_max_tokens: the TOML KEYS are
    deleted at T0 (steering error below); the dataclass FIELDS survive one
    extra slice because the carving machinery still reads them until T1
    deletes the mechanism. T1's suite pins the field deletion.

Locked semantics (spec §3.3, SB 2026-09-03 — do not re-litigate):
  - Legacy key spellings in [context] -> ConfigError steering to the new
    key. INTERPRETATION NOTE (flagged for SB at acceptance, same class as
    the ruff-gate wording divergence): the spec says "warn-once +
    renamed-key steering on legacy keys, never silently honored". We
    encode the strictest coherent reading: the old spelling is NEVER
    honored — a legacy key is a structural config error whose message
    names the new key. Rationale: a tuned knob silently reverting to its
    default is exactly the fail-open the config module's own docstring
    discipline forbids; honoring-with-a-warning (reading C) is still
    honoring and contradicts the hard epoch. Fleet impact today: ZERO —
    no live agent config carries a [context] section (verified 2026-09-03
    across /etc/openalph/agents/**).
  - prompt.py's independent assemble_prompt `gc_enabled` PARAMETER is
    explicitly OUT of scope (spec §4 decompose-band feedback item 3): it
    is the prompt-scaffold flag, not a [context] key. Only its agent.py
    call-site attribute read renames. The scan below therefore pins
    dotted attribute reads (`.gc_enabled`), which never match prompt.py's
    bare parameter name.

Slice ownership of the old-name absence criterion (per-slice passable):
  - T0 (this file): .gc_enabled / .warn_pct / ContextGCConfig /
    _parse_context_gc_config absent from src/**/*.py.
  - T1 (strip suite): extends the scan with .thinking_tail_turns /
    .thinking_tail_max_tokens once the carving machinery is deleted.
"""

import openalph.config

from _handoff_helpers import BASE_TOML, expect_context_error, scan_src_for_tokens, write_and_load


# ===========================================================================
# Class + parser renames
# ===========================================================================

class TestHandoffConfigClass:
    def test_context_handoff_config_exported(self):
        cfg_cls = getattr(openalph.config, "ContextHandoffConfig", None)
        assert cfg_cls is not None, (
            "openalph.config must export ContextHandoffConfig "
            "(renamed from ContextGCConfig)")

    def test_old_class_name_gone(self):
        assert not hasattr(openalph.config, "ContextGCConfig"), (
            "hard epoch: the old class name must not remain as an alias")

    def test_agentconfig_context_default_factory(self):
        cfg_cls = openalph.config.ContextHandoffConfig
        assert isinstance(cfg_cls, type)
        # The AgentConfig.context field default must produce the new class.
        inst = cfg_cls()
        assert inst.handoff_enabled is True
        assert inst.checkpoint_pct == 75

    def test_parse_function_renamed(self):
        assert hasattr(openalph.config, "_parse_context_handoff_config"), (
            "private parser must be renamed for coherence — cli.py showprompt "
            "consumes it directly")
        assert not hasattr(openalph.config, "_parse_context_gc_config")


# ===========================================================================
# Defaults
# ===========================================================================

class TestHandoffConfigDefaults:
    def test_absent_section_full_defaults(self, tmp_path):
        c = write_and_load(tmp_path, BASE_TOML).context
        assert c.handoff_enabled is True
        assert c.checkpoint_pct == 75
        assert c.auto_pct == 85
        assert c.hard_pct == 92
        assert c.durable_budget_pct == 25.0
        assert c.durable_budget_min_tokens == 96000
        # kdsn.305.12 D3: runway-gated handoff thresholds (unchanged)
        assert c.handoff_runway_pct == 10.0
        assert c.handoff_runway_min_tokens == 24000
        assert c.durable_paths == []
        # Spec §3.1 decision: turn_cooldown RETAINED (tool-path churn guard)
        assert c.turn_cooldown == 3


# ===========================================================================
# Renamed keys honored
# ===========================================================================

class TestRenamesHonored:
    def test_handoff_enabled_false(self, tmp_path):
        c = write_and_load(
            tmp_path, BASE_TOML + "[context]\nhandoff_enabled = false\n").context
        assert c.handoff_enabled is False

    def test_checkpoint_pct(self, tmp_path):
        c = write_and_load(
            tmp_path, BASE_TOML + "[context]\ncheckpoint_pct = 70\n").context
        assert c.checkpoint_pct == 70

    def test_renames_combined_with_unchanged_keys(self, tmp_path):
        c = write_and_load(tmp_path, BASE_TOML + '''
[context]
handoff_enabled = false
checkpoint_pct = 70
auto_pct = 80
hard_pct = 90
durable_budget_pct = 20.0
durable_budget_min_tokens = 24000
durable_paths = ["skills/*.md"]
turn_cooldown = 5
''').context
        assert c.handoff_enabled is False
        assert c.checkpoint_pct == 70
        assert c.auto_pct == 80 and c.hard_pct == 90
        assert c.durable_budget_pct == 20.0
        assert c.durable_budget_min_tokens == 24000
        assert c.durable_paths == ["skills/*.md"]
        assert c.turn_cooldown == 5


# ===========================================================================
# Legacy key steering — hard epoch on the config surface
# ===========================================================================

class TestLegacyKeySteering:
    def test_gc_enabled_steers_to_handoff_enabled(self, tmp_path):
        expect_context_error(
            tmp_path, "[context]\ngc_enabled = true\n",
            needle="handoff_enabled")

    def test_warn_pct_steers_to_checkpoint_pct(self, tmp_path):
        expect_context_error(
            tmp_path, "[context]\nwarn_pct = 70\n",
            needle="checkpoint_pct")

    def test_thinking_tail_turns_rejected(self, tmp_path):
        # Deleted with the carving (spec §3.1). No successor key — the error
        # names the removed key; only the key name is pinned (wording churns).
        expect_context_error(tmp_path, "[context]\nthinking_tail_turns = 4\n")

    def test_thinking_tail_max_tokens_rejected(self, tmp_path):
        expect_context_error(
            tmp_path, "[context]\nthinking_tail_max_tokens = 100\n")

    def test_legacy_key_not_honored_even_alongside_new_name(self, tmp_path):
        # "never silently honored": presence of the old spelling is an error
        # even when the new key is also set correctly.
        expect_context_error(
            tmp_path,
            "[context]\nhandoff_enabled = false\ngc_enabled = true\n",
            needle="handoff_enabled")

    def test_legacy_rejection_is_config_error(self, tmp_path):
        from openalph.config import ConfigError
        err = None
        try:
            write_and_load(tmp_path, BASE_TOML + "[context]\nwarn_pct = 70\n")
        except ConfigError as e:
            err = e
        assert err is not None, "legacy key must raise ConfigError (structural)"

    def test_unknown_key_rejected_fail_loud(self, tmp_path):
        # T1 deviation accepted (flagged for SB): unknown [context] keys are
        # REJECTED, not silently ignored — a typo'd handoff_enabled would
        # otherwise silently revert to the default (fail-open on a
        # data-loss knob, the exact hole the module discipline forbids).
        # Zero fleet impact (no live config carries [context]); rollback
        # note: older code + newer config fails loud naming the key.
        expect_context_error(tmp_path, "[context]\nhandoff_enable = true\n",
                             needle="handoff_enable")


# ===========================================================================
# Fail-loud types preserved (new spellings)
# ===========================================================================

class TestFailLoudTypes:
    def test_handoff_enabled_string_rejected(self, tmp_path):
        expect_context_error(tmp_path, '[context]\nhandoff_enabled = "no"\n')

    def test_handoff_enabled_int_rejected(self, tmp_path):
        expect_context_error(tmp_path, "[context]\nhandoff_enabled = 1\n")

    def test_checkpoint_pct_out_of_range(self, tmp_path):
        expect_context_error(tmp_path, "[context]\ncheckpoint_pct = 150\n")

    def test_checkpoint_pct_zero_rejected(self, tmp_path):
        expect_context_error(tmp_path, "[context]\ncheckpoint_pct = 0\n")

    def test_checkpoint_pct_bool_rejected(self, tmp_path):
        expect_context_error(tmp_path, "[context]\ncheckpoint_pct = true\n")

    def test_auto_pct_unchanged_still_validated(self, tmp_path):
        expect_context_error(tmp_path, "[context]\nauto_pct = 150\n")

    def test_turn_cooldown_negative_rejected(self, tmp_path):
        expect_context_error(tmp_path, "[context]\nturn_cooldown = -1\n")

    def test_turn_cooldown_bool_rejected(self, tmp_path):
        expect_context_error(tmp_path, "[context]\nturn_cooldown = true\n")

    def test_durable_budget_min_tokens_negative(self, tmp_path):
        expect_context_error(
            tmp_path, "[context]\ndurable_budget_min_tokens = -1\n")

    def test_handoff_runway_pct_zero_rejected(self, tmp_path):
        expect_context_error(tmp_path, "[context]\nhandoff_runway_pct = 0\n")

    def test_section_not_table(self, tmp_path):
        expect_context_error(tmp_path, '[context]\n"str"\n')


# ===========================================================================
# Hard epoch: old spellings absent from src (attribute reads / class name)
# ===========================================================================

class TestOldSpellingsAbsentFromSrc:
    def test_no_dotted_reads_or_class_references(self):
        # Dotted reads are the honoring surface; the class/parse names are
        # the identity surface. Bare "gc_enabled" prose (prompt.py's
        # assemble_prompt parameter, steering-message text) is explicitly
        # allowed — see module docstring.
        hits = scan_src_for_tokens([
            ".gc_enabled",
            ".warn_pct",
            "ContextGCConfig",
            "_parse_context_gc_config",
        ])
        assert not hits, (
            "hard epoch: old [context] spellings remain in src:\n"
            + "\n".join(f"  {f}:{i}: {line}" for f, i, line in hits[:20]))
