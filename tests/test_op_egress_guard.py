"""RED/round-2 suite — L2 shell egress guard (op-handling design).

Originally authored from op-handling-test-plan.md; round-2 revises the
predicate to a presence-based algorithm per remediation-spec-r2.md §A
(structural rewrite — see tools/security.py for the algorithm). TESTS ONLY.

    tools/security.py :: op_egress_block_reason(command: str) -> str | None
    tools/shell.py    :: run_shell gate calling op_egress_block_reason at top

Predicate contract: returns the redirect message (containing the verbatim
substrings "op-run" and "op inject") when `command` would print a 1Password
secret to stdout (→ context); else None. Pure, no I/O.

BLOCK: `op read <ref>`, `opread <ref>`, `op document get <ref>`, `op item get
… --fields|--format|--reveal|--otp`, any of the above as a top-level segment
of a `;`/`&&`/`||`/`&`/`|`/newline-separated sequence — INCLUDING as the last
stage of a pipe (round-2: pipe-allow is dropped; a piped op-egress command
now BLOCKS, since the piped consumer's stdout can still resurface in
context) — wrapped in `sudo`/`env`/`time`/`nohup`/an absolute path/an
env-var assignment/a boolean global `op` flag (`--debug`, `--no-color`), or
inside a subshell `(...)`/brace `{...}`/`if` group (presence-based: the
guard scans for op-egress keyword tokens anywhere in a top-level segment,
not a fixed wrapper-stripping/positional-binary shape).

ALLOW: `op item get` without a dumping flag (masked overview), management
(`op item create/edit/list`, `op vault …`, `op inject`, `op run`,
`op signin/whoami/account`), `opsudo …`, any non-op command, `op read` only
inside a quoted arg of another command (command-position-anchored, not
arbitrary substring), and — the dominant safe pattern — a value captured via
command substitution, `VAR=$(opread "op://…")` / `VAR=$(op read "op://…")`
(the value lands in a shell variable, never directly in stdout/context; the
guard excises `$(...)` spans before scanning, per spec §A step 4).

RED honesty:
  * predicate unit cases import op_egress_block_reason INSIDE the test body →
    a legible per-case ImportError (the correct "feature absent" RED), never a
    module-collection error.
  * the BLOCK executor tests (#21/#22) are behavioral RED: with no guard the
    real `op`/`opread` subprocess runs and its output (an auth error — no
    OP_SERVICE_ACCOUNT_TOKEN in the test env, fake refs) lands in content, so
    the redirect-message substrings are ABSENT → AssertionError.  Fake op://
    refs only; verified to fail fast (exit 1, no hang, no network).
  * ALLOW / stateless executor tests reference the new predicate so they are
    RED-now via ImportError while remaining faithful once green (deviation #3).
  * round-2: the moved pipe-allow→block params (and the revised test #23)
    are RED-confirmed-before-fix against the ROUND-1 implementation still on
    disk at the time the test file is first deployed (see remediation-report-r2.md).
"""

import logging

import pytest

from openalph.config import AgentConfig, ProviderConfig
from openalph.tools import ToolResult, execute_tool


# --- New-symbol loader (import-inside-test → legible per-test RED) ----------

def _load_op_egress():
    from openalph.tools.security import op_egress_block_reason
    return op_egress_block_reason


@pytest.fixture(autouse=True)
def _clean_api_key_cache():
    """Snapshot + restore _api_key_cache so no L2 test can leak module state."""
    from openalph.tools import _api_key_cache
    snapshot = dict(_api_key_cache)
    try:
        yield
    finally:
        _api_key_cache.clear()
        _api_key_cache.update(snapshot)


def _make_config(tmp_path):
    return AgentConfig(
        name="test-op-egress",
        default_model="anthropic/claude-sonnet-4-20250514",
        max_tokens=8192,
        providers={"anthropic": ProviderConfig(
            key="anthropic", type="anthropic", api_key="sk-test",
            base_url=None, quirks=[])},
        workspace=tmp_path,
    )


SHELL_TOOL_CONFIG = {"default_timeout": 30, "max_output": 50000}


# ===========================================================================
# Unit predicate — BLOCK cases  (spec cases 1–9)
# Assert: returns non-None message containing "op-run" and "op inject".
# ===========================================================================

BLOCK_CASES = [
    pytest.param('op read "op://vault/item/field"', id="op_read_quoted"),
    pytest.param('op read op://vault/item/field', id="op_read_unquoted"),
    pytest.param('op   read    "op://x"', id="op_read_extra_whitespace"),
    pytest.param('opread "op://vault/item/field"', id="opread_bare"),
    pytest.param('op document get "My Doc"', id="op_document_get"),
    pytest.param('op item get "Item" --fields password', id="op_item_get_fields"),
    pytest.param('op item get "Item" --format json', id="op_item_get_format"),
    pytest.param('op item get "Item" --reveal', id="op_item_get_reveal"),
    pytest.param('   op read "op://x"', id="op_read_leading_whitespace"),
    pytest.param('op read "op://x" || echo failed', id="op_read_or_echo"),
    pytest.param('op read "op://x" || true', id="op_read_or_true"),
    pytest.param('cd /app && op read "op://x/y"', id="cd_and_op_read"),
    pytest.param('echo starting; op read "op://x/y"', id="echo_semi_op_read"),
    pytest.param('true && op read "op://x"', id="true_and_op_read"),
    pytest.param('echo x | op read "op://y"', id="op_read_last_pipe_stage"),
    pytest.param('op item get "Item" --otp', id="op_item_get_otp"),
    pytest.param('sudo op read "op://x"', id="sudo_op_read"),
    pytest.param('env op read "op://x"', id="env_op_read"),
    pytest.param('/usr/bin/op read "op://x"', id="abs_path_op_read"),
    pytest.param('op --account prod read op://vault/item/field', id="op_global_flag_read"),
    pytest.param('OP_ACCOUNT=prod op read op://vault/item/field', id="op_env_assign_read"),
    pytest.param('cd /app\nop read "op://vault/db/password"', id="multiline_op_read"),
    # -- moved from ALLOW (round-2: pipe-allow dropped, spec §A step 5/7) --
    pytest.param('op read "op://x" | jq .', id="op_read_piped_jq"),
    pytest.param('opread "op://x" | sudo -S tee /etc/foo', id="opread_piped_sudo"),
    pytest.param('op read "op://x" | cat', id="op_read_piped_cat"),
    # -- round-2 additions (H-1 merged punctuation + presence-based corners) --
    pytest.param('cd /app &&\nop read "op://vault/db/password"', id="op_read_after_andand_newline"),
    pytest.param('echo done ;\nop read "op://x"', id="op_read_after_semi_newline"),
    pytest.param('false ||\nop read "op://x"', id="op_read_after_oror_newline"),
    pytest.param('echo x &\nop read "op://x"', id="op_read_after_amp_newline"),
    pytest.param('op --debug read op://x', id="op_boolean_global_flag_read"),
    pytest.param('op --no-color read op://x', id="op_nocolor_flag_read"),
    pytest.param('sudo -u root op read "op://x"', id="sudo_u_value_op_read"),
    pytest.param('env OP_ACCOUNT=prod op read "op://x"', id="env_assign_after_env_op_read"),
    pytest.param('env -i OP_ACCOUNT=x op read "op://x"', id="env_i_assign_op_read"),
    pytest.param('time op read "op://x"', id="time_op_read"),
    pytest.param('nohup op read "op://x"', id="nohup_op_read"),
    pytest.param('/usr/bin/env op read "op://x"', id="abs_env_op_read"),
    pytest.param('(op read "op://x")', id="subshell_group_op_read"),
    pytest.param('{ op read "op://x"; }', id="brace_group_op_read"),
    pytest.param('if true; then op read "op://x"; fi', id="if_then_op_read"),
    pytest.param('op read "op://x" >&2 | cat', id="op_read_redirect_stderr_pipe"),
    pytest.param('op --format=json item get Item', id="op_global_format_item_get"),
    # -- round-3 additions (H1: fail-closed command-substitution excision) --
    pytest.param("x=$(printf '('); op read op://x", id="cmdsub_solo_paren_then_op_read"),
    pytest.param("echo '<('; op read op://x", id="literal_procsub_then_op_read"),
    # deviation (see remediation-report-r3.md): spec's literal "match to the
    # FIRST following )" wording, read as a single left-to-right sequential
    # scan with no stack, would let this unmatched '<(' decoy's search reach
    # PAST the real `op read` and consume the `)` that belongs to the later,
    # unrelated, legitimate `$(true)` span -- excising the real op-read
    # along with it and reintroducing a fail-open. A LIFO-stack match
    # (each `)` closes the MOST RECENTLY opened still-open genuine opener)
    # gives identical results on every spec example while also closing this
    # gap: the decoy `<(` never has a genuine opener to its right that it
    # can "steal", so it stays unmatched -> unexcised -> `op read` stays
    # scannable -> BLOCK.
    pytest.param("echo '<('; op read op://x; V=$(true)", id="literal_procsub_before_op_read_and_later_cmdsub"),
    # -- round-3 additions (H2: backslash-newline continuation pre-pass) --
    pytest.param("op \\\nread op://x", id="backslash_newline_before_read"),
    pytest.param("op item get Item \\\n--fields password", id="backslash_newline_before_fields"),
    pytest.param("op \\\ndocument get Doc", id="backslash_newline_before_document"),
    # -- round-3 addition (M-1 ambiguous-flag safety net regression case) --
    # `--account`'s VALUE ("item") coincidentally spells a known-subcommand
    # keyword; the REAL subcommand is `read`, one token further along. A
    # naive "first known-keyword wins" scan (the literal M-1 spec
    # algorithm with no refinement) stops at "item" and misses the real
    # `read` -- found empirically via the differential harness
    # (verify_guard.py) during implementation, see remediation-report-r3.md.
    pytest.param("op --account item read op://x", id="account_value_collides_with_subcommand_keyword"),
    # -- round-4 additions (H1: quoted-decoy '<(' / '$' '(' opens a synthetic
    # excision span a later literal ')' closes; provably-fail-closed rewrite) --
    pytest.param("echo '<('; op read op://x; echo ')'", id="quoted_procsub_later_close_read"),
    pytest.param("echo '<('; op document get Doc; echo ')'", id="quoted_procsub_later_close_docget"),
    pytest.param("echo '<('; op item get Item --fields password; echo ')'", id="quoted_procsub_later_close_itemfields"),
    pytest.param("echo '<('; opread op://x; echo ')'", id="quoted_procsub_later_close_opread"),
    pytest.param("echo '$' '('; op read op://x; echo ')'", id="quoted_dollar_paren_later_close_read"),
    # -- round-4 additions (H2: --config/--session value-taking globals) --
    pytest.param("op --config item read op://x", id="config_value_item_hides_read"),
    pytest.param("op --config run read op://x", id="config_value_run_hides_read"),
    pytest.param("op --config run item get Item --fields password", id="config_value_run_hides_itemfields"),
    pytest.param("op --session x read op://x", id="session_value_hides_read"),
    # -- round-4 additions (H3: singular --field dump-flag alias) --
    pytest.param("op item get Item --field password", id="item_get_singular_field"),
    pytest.param("op item get Item --field=password", id="item_get_singular_field_eq"),
    # ======================================================================
    # round-5 additions -- the deny-by-default rewrite (SEC-1/2/3).
    #
    # 34 of the 36 cases below were confirmed to ALLOW against the round-4
    # implementation before the fix, via a differential harness run over
    # both versions (see the PR description for the transcript); each of
    # those is a live 1Password secret reaching model context through the
    # control that exists to prevent exactly that.
    #
    # The two exceptions are kept deliberately, marked inline: they already
    # blocked under round-4 by accident of the old algorithm, and they are
    # retained as regression guards so the rewrite cannot lose coverage the
    # previous implementation happened to have.
    # ======================================================================
    # -- SEC-1: `$(...)` in a NON-capture position prints to stdout. Round-4
    #    excised any `$(...)` whose opener followed a `$`, on the premise
    #    that the value "lands in a shell variable" -- true only for an
    #    assignment, not for a command's argument.
    pytest.param("echo $(op read op://vault/item)", id="r5_echo_cmdsub_op_read"),
    pytest.param("printf %s $(op read op://x)", id="r5_printf_cmdsub_op_read"),
    pytest.param("echo $(opread op://x)", id="r5_echo_cmdsub_opread"),
    pytest.param("echo $(op document get Doc)", id="r5_echo_cmdsub_document_get"),
    pytest.param("echo $(op item get Item --fields password)", id="r5_echo_cmdsub_item_fields"),
    pytest.param("echo $(  op   read   op://x  )", id="r5_cmdsub_inner_whitespace"),
    pytest.param('echo "prefix $(op read op://x) suffix"', id="r5_cmdsub_interpolated_in_string"),
    pytest.param("echo $(echo $(op read op://x))", id="r5_cmdsub_nested"),
    pytest.param("echo x > $(op read op://x)", id="r5_cmdsub_in_redirect_target"),
    pytest.param("A=$(true) echo $(op read op://x)", id="r5_capture_then_noncapture_cmdsub"),
    pytest.param("foo=$(a)$(op read op://x)", id="r5_concatenated_cmdsub_not_capture"),
    # (already blocked under round-4 -- no closing `)` meant nothing was
    #  excised, so the tokens stayed scannable; kept as a regression guard)
    pytest.param("echo $(op read op://x", id="r5_unterminated_cmdsub"),
    # -- SEC-1b: a DOUBLE-QUOTED substitution collapses to one shlex token
    #    whose basename is neither `op` nor `opread`, so a token-level scan
    #    never saw inside it. Found while fixing SEC-1; not in the report.
    pytest.param('echo "$(op read op://x)"', id="r5_quoted_cmdsub_op_read"),
    pytest.param('echo "${x}$(op read op://x)"', id="r5_quoted_cmdsub_after_param"),
    # -- SEC-2: backticks. shlex treats ` as an ordinary word character, so
    #    `` echo `op read x` `` tokenized as ['echo', '`op', 'read', 'op://x`']
    #    and no basename ever matched. Backticks print to stdout exactly as
    #    `$(...)` does.
    pytest.param("echo `op read op://x`", id="r5_backtick_op_read"),
    pytest.param('echo "`op read op://x`"', id="r5_quoted_backtick_op_read"),
    pytest.param("cat <<< `op read op://x`", id="r5_backtick_herestring"),
    pytest.param("echo `opread op://x`", id="r5_backtick_opread"),
    pytest.param("echo `echo \\`op read op://x\\``", id="r5_backtick_nested_escaped"),
    pytest.param("echo `op read op://x", id="r5_unterminated_backtick"),
    # -- SEC-3: a command runner's quoted argument is opaque to a token
    #    scan. `sh -c 'op read op://x'` emitted the inner command as ONE
    #    token whose basename is `x` (from `op://x`), matching nothing.
    pytest.param("sh -c 'op read op://x'", id="r5_sh_c_op_read"),
    pytest.param('bash -c "op read op://x"', id="r5_bash_c_op_read"),
    pytest.param("/bin/sh -c 'op read op://x'", id="r5_abs_sh_c_op_read"),
    pytest.param("bash -lc 'op read op://x'", id="r5_bash_bundled_flag_op_read"),
    pytest.param("zsh -c 'op read op://x'", id="r5_zsh_c_op_read"),
    pytest.param("dash -c 'op read op://x'", id="r5_dash_c_op_read"),
    pytest.param("busybox sh -c 'op read op://x'", id="r5_busybox_sh_c_op_read"),
    pytest.param("eval 'op read op://x'", id="r5_eval_quoted_op_read"),
    # (already blocked under round-4 -- unquoted, so `op`/`read` were plain
    #  tokens the presence scan saw; kept as a regression guard)
    pytest.param("eval op read op://x", id="r5_eval_bare_op_read"),
    pytest.param("env FOO=1 sh -c 'op read op://x'", id="r5_env_sh_c_op_read"),
    pytest.param("sudo sh -c 'op read op://x'", id="r5_sudo_sh_c_op_read"),
    pytest.param("xargs -I{} sh -c 'op read op://x'", id="r5_xargs_sh_c_op_read"),
    pytest.param("timeout 5 sh -c 'op read op://x'", id="r5_timeout_sh_c_op_read"),
    pytest.param("python3 -c 'op read op://x'", id="r5_code_flag_generic"),
    pytest.param("sh -c 'sh -c \"op read op://x\"'", id="r5_sh_c_nested"),
    pytest.param('sh -c "echo $(op read op://x)"', id="r5_sh_c_wrapping_cmdsub"),
    # -- round-5 follow-up: the assignment-capture carve-out only trusts a
    #    capture in COMMAND POSITION. `word=$(op read x)` as an ARGUMENT
    #    prints to stdout -- its left-context ends in `NAME=` like a real
    #    capture, but a plain space is not command position. Found by
    #    adversarial review after the first round-5 pass; all leak in a real
    #    shell (verified with `op` stubbed to `echo SECRET`).
    pytest.param("echo NAME=$(op read op://x)", id="r5b_arg_assign_cmdsub"),
    pytest.param('echo X="$(op read op://x)"', id="r5b_arg_assign_quoted_cmdsub"),
    pytest.param("true && echo Y=$(op read op://x)", id="r5b_arg_assign_after_and"),
    pytest.param("printf tok=$(op read op://x)", id="r5b_arg_assign_printf"),
    pytest.param("curl -d TOKEN=$(op read op://x) http://h", id="r5b_arg_assign_curl_d"),
    pytest.param("echo tok=`op read op://x`", id="r5b_arg_assign_backtick"),
    pytest.param("sh -c 'echo Z=$(op read op://x)'", id="r5b_arg_assign_in_sh_c"),
    pytest.param("cmd arg=$(op read op://x)", id="r5b_arg_assign_generic"),
    pytest.param("echo pre NAME=$(op read op://x)", id="r5b_arg_assign_two_words"),
    pytest.param("echo -x N=$(op read op://x)", id="r5b_flag_then_arg_assign"),
]


class TestEgressPredicateBlock:

    @pytest.mark.parametrize("command", BLOCK_CASES)
    def test_blocks_egress_command(self, command):
        op_egress_block_reason = _load_op_egress()
        reason = op_egress_block_reason(command)
        assert reason is not None, f"expected BLOCK for {command!r}, got None"
        assert isinstance(reason, str)
        assert "op-run" in reason, "redirect message must name the safe path 'op-run'"
        assert "op inject" in reason, "redirect message must mention 'op inject'"


# ===========================================================================
# Unit predicate — ALLOW cases  (spec cases 10–20)
# Assert: returns None.  ("/"-joined sub-cases split into distinct params.)
# ===========================================================================

ALLOW_CASES = [
    pytest.param('op item get "Item"', id="op_item_get_masked_no_flag"),
    pytest.param('op item create --title X', id="op_item_create"),
    pytest.param('op item edit X', id="op_item_edit"),
    pytest.param('op item list', id="op_item_list"),
    pytest.param('op vault list', id="op_vault_list"),
    pytest.param('op inject -i template.tpl -o out.conf', id="op_inject"),
    pytest.param('op run -- curl https://api', id="op_run"),
    pytest.param('op signin', id="op_signin"),
    pytest.param('op whoami', id="op_whoami"),
    pytest.param('op account get', id="op_account_get"),
    pytest.param('opsudo apt-get update', id="opsudo_management"),
    pytest.param('echo hello', id="non_op_echo"),
    pytest.param('echo "op read is a thing"', id="op_read_inside_quoted_arg"),
    pytest.param('', id="empty_string"),
    pytest.param('echo "hi; op read x"', id="op_read_in_quoted_semicolon"),
    pytest.param('op item get "foo" --show-all-fields', id="dump_flag_substring_not_flag"),
    pytest.param('op item get "Item --format"', id="dump_flag_inside_quoted_name"),
    pytest.param('op readme', id="op_readme_not_read"),
    pytest.param('opreadiness check', id="opreadiness_not_opread"),
    # -- round-2 additions: the dominant safe $(...) capture pattern + regressions --
    pytest.param('SHODAN_KEY=$(opread "op://x")', id="var_cmdsub_opread"),
    pytest.param('export K=$(op read "op://x")', id="export_cmdsub_op_read"),
    pytest.param('PRIVKEY=$(opread "op://shelly/Merry Ethereum Wallet/private key")', id="cmdsub_opread_realistic"),
    pytest.param('MSMTP_PASS=$(opread "op://v/migadu/password") msmtp foo', id="cmdsub_then_use"),
    # -- round-3 additions (M-1: precise op-subcommand detection) --
    pytest.param("op run -- make read", id="op_run_child_named_read"),
    pytest.param("op inject -i read -o out.conf", id="op_inject_template_named_read"),
    pytest.param("op item get read", id="op_item_get_item_named_read"),
    pytest.param("op item create --title read", id="op_item_create_title_read"),
    pytest.param('DB=$(opread "op://v/db/password"); psql', id="cmdsub_opread_then_use"),
    # -- round-4 additions (regression: fixes must not over-block these) --
    pytest.param('V=$(opread "op://x")', id="cmdsub_opread_still_allow"),
    pytest.param('V=$(op read "op://x")', id="cmdsub_op_read_still_allow"),
    pytest.param("op item get Item --show-all-fields", id="field_substring_not_flag_still_allow"),
    pytest.param("op run -- make read", id="op_run_child_read_still_allow"),
    # -- round-5 additions: the deny-by-default rewrite must not over-block --
    # The capture carve-out is the ONE trusted shape and must survive in all
    # of its spellings, including the quoted and backtick forms the SEC-1/2
    # fixes newly parse.
    pytest.param('V="$(op read op://x)"', id="r5_quoted_capture_still_allow"),
    pytest.param("V=`op read op://x`", id="r5_backtick_capture_still_allow"),
    pytest.param("arr[0]=$(op read op://x)", id="r5_array_subscript_capture"),
    pytest.param("local V=$(opread op://x)", id="r5_local_capture"),
    pytest.param("A=1 B=$(op read op://x) cmd", id="r5_env_prefix_capture"),
    # A SINGLE-quoted substitution is a literal string to the shell -- no
    # substitution happens, so nothing is printed and nothing is egress.
    pytest.param("V='$(op read op://x)'", id="r5_single_quoted_literal_not_subst"),
    # Ordinary substitutions and `-c` flags on non-shell commands must not
    # be collateral damage of the recursion.
    pytest.param("echo $(date)", id="r5_benign_cmdsub"),
    pytest.param("D=$(date); echo $D", id="r5_benign_capture_then_use"),
    pytest.param("echo $((1+2))", id="r5_arithmetic_expansion"),
    pytest.param("grep -c 'foo(' f.txt", id="r5_grep_c_flag_not_shell_code"),
    pytest.param('git commit -m "wip"', id="r5_ordinary_command"),
    # -- round-5 follow-up: real captures in command position, including the
    #    declaration-keyword and multi-assignment forms, must still ALLOW
    #    after the carve-out was tightened to require command position.
    pytest.param("readonly R=$(op read op://x)", id="r5b_readonly_capture"),
    pytest.param("declare -x D=$(op read op://x)", id="r5b_declare_flag_capture"),
    pytest.param("A=1 B=$(op read op://x) cmd", id="r5b_multi_assign_capture"),
    pytest.param("  VAR=$(op read op://x)", id="r5b_leading_ws_capture"),
    pytest.param("local V=$(op read op://x)", id="r5b_local_capture_still"),
]


class TestEgressPredicateAllow:

    @pytest.mark.parametrize("command", ALLOW_CASES)
    def test_allows_command(self, command):
        op_egress_block_reason = _load_op_egress()
        reason = op_egress_block_reason(command)
        assert reason is None, f"expected ALLOW (None) for {command!r}, got {reason!r}"


# ===========================================================================
# Executor integration — guard short-circuits before subprocess  (tests 21–25)
# execute_tool / run_shell are NOT mocked — the guard is the code under test.
# ===========================================================================

class TestExecutorEgressGuard:

    @pytest.mark.asyncio
    async def test_execute_tool_blocks_op_read(self, tmp_path):
        """21: `op read` via execute_tool → is_error + redirect message; no op output.

        RED (no guard): the real `op read` subprocess runs and its auth error —
        NOT the redirect message — lands in content, so the substring asserts fail.
        """
        cfg = _make_config(tmp_path)
        result = await execute_tool(
            name="shell",
            input={"command": 'op read "op://x/y"'},
            tool_config=SHELL_TOOL_CONFIG,
            agent_config=cfg,
        )
        assert isinstance(result, ToolResult)
        assert result.is_error is True
        assert "op-run" in result.content, \
            "blocked op read must return the redirect message (guard ran before subprocess)"
        assert "op inject" in result.content
        # The op subprocess must NOT have run: its auth-failure text must be absent.
        assert "No accounts configured" not in result.content
        assert "OP_SERVICE_ACCOUNT_TOKEN" not in result.content

    @pytest.mark.asyncio
    async def test_execute_tool_blocks_bare_opread(self, tmp_path):
        """22: bare `opread "op://x"` via execute_tool → is_error + redirect message."""
        cfg = _make_config(tmp_path)
        result = await execute_tool(
            name="shell",
            input={"command": 'opread "op://x"'},
            tool_config=SHELL_TOOL_CONFIG,
            agent_config=cfg,
        )
        assert isinstance(result, ToolResult)
        assert result.is_error is True
        assert "op-run" in result.content
        assert "op inject" in result.content

    @pytest.mark.asyncio
    async def test_execute_tool_blocks_piped_opread_but_allows_benign_pipe(self, tmp_path):
        """23 (round-2 revised): pipe-allow is dropped — a piped opread form is
        now BLOCKED (pipe is just another segment separator, spec §A step 5/7)
        — while an unrelated benign pipe with no op-egress token still runs
        normally (the guard is presence-based per SEGMENT, not "pipes are
        blanket-blocked").

        (Round-1's `test_execute_tool_allows_piped_opread` asserted the
        opposite — `opread "op://x" | cat"` was allowed — because pipe-allow
        was still in effect; round-2 drops pipe-allow entirely, so that
        assertion is now stale and is corrected here rather than left
        contradicting the spec.)
        """
        op_egress_block_reason = _load_op_egress()
        # Predicate: piped opread → now BLOCKED.
        reason = op_egress_block_reason('opread "op://x" | cat')
        assert reason is not None
        assert "op-run" in reason
        assert "op inject" in reason
        # Executor: an unrelated benign piped command still runs normally.
        cfg = _make_config(tmp_path)
        result = await execute_tool(
            name="shell",
            input={"command": "echo piped-through | cat"},
            tool_config=SHELL_TOOL_CONFIG,
            agent_config=cfg,
        )
        assert result.is_error is False
        assert "piped-through" in result.content

    @pytest.mark.asyncio
    async def test_execute_tool_allows_normal_command(self, tmp_path):
        """24: a normal command (`echo hi`) runs and is not blocked."""
        op_egress_block_reason = _load_op_egress()
        assert op_egress_block_reason("echo hi") is None
        cfg = _make_config(tmp_path)
        result = await execute_tool(
            name="shell",
            input={"command": "echo hi"},
            tool_config=SHELL_TOOL_CONFIG,
            agent_config=cfg,
        )
        assert result.is_error is False
        assert "hi" in result.content
        # A benign echo must never be mistaken for the redirect message.
        assert "op-run" not in result.content

    def test_block_is_stateless(self, tmp_path):
        """25: the predicate mutates no module state — identical result on repeat,
        _api_key_cache untouched."""
        op_egress_block_reason = _load_op_egress()
        from openalph.tools import _api_key_cache
        before = dict(_api_key_cache)
        cmd = 'op read "op://x/y"'
        r1 = op_egress_block_reason(cmd)
        r2 = op_egress_block_reason(cmd)
        assert r1 == r2
        assert r1 is not None
        assert dict(_api_key_cache) == before, \
            "op_egress_block_reason must not mutate _api_key_cache"

    @pytest.mark.asyncio
    async def test_non_string_command_does_not_raise(self, tmp_path):
        # A malformed (non-str) command must not raise out of the guard.
        from openalph.tools.security import op_egress_block_reason
        assert op_egress_block_reason(["op", "read", "op://x"]) is None
        assert op_egress_block_reason(None) is None
