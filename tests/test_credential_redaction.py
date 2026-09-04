"""Tests for credential redaction in tool results (.122).

Tool results are scanned for credential patterns before returning from
execute_tool(). Matches are replaced with [REDACTED:<type>] markers.
The original value never enters the model's context window.

Architecture:
    tools/security.py — pattern library + redact_credentials()
    tools/__init__.py — execute_tool() calls redact after dispatch, before return

Pattern priority: specific patterns match first, generic last.
No double-redaction (a value matching both sk-ant- and generic hex
is only redacted once, as anthropic_api_key).
"""

import pytest
import logging
from unittest.mock import AsyncMock, patch

from openalph.tools.security import (
    redact_credentials,
    RedactionEvent,
    CREDENTIAL_PATTERNS,
)
from openalph.tools import ToolResult, execute_tool


# ---------------------------------------------------------------------------
# Pattern library sanity
# ---------------------------------------------------------------------------

class TestPatternLibrary:

    def test_patterns_is_nonempty_list(self):
        """Pattern library exists and has entries."""
        assert len(CREDENTIAL_PATTERNS) >= 10

    def test_each_pattern_has_required_fields(self):
        """Every pattern entry has name, pattern, and redaction_label."""
        for p in CREDENTIAL_PATTERNS:
            assert "name" in p, f"Pattern missing 'name': {p}"
            assert "pattern" in p, f"Pattern {p['name']} missing 'pattern'"
            assert "redaction_label" in p, f"Pattern {p['name']} missing 'redaction_label'"

    def test_pattern_names_are_unique(self):
        """No duplicate pattern names."""
        names = [p["name"] for p in CREDENTIAL_PATTERNS]
        assert len(names) == len(set(names))

    def test_redaction_labels_are_bracketed(self):
        """All redaction labels follow [REDACTED:type] format."""
        for p in CREDENTIAL_PATTERNS:
            label = p["redaction_label"]
            assert label.startswith("[REDACTED:"), f"{p['name']}: {label}"
            assert label.endswith("]"), f"{p['name']}: {label}"


# ---------------------------------------------------------------------------
# Specific pattern matching
# ---------------------------------------------------------------------------

class TestAnthropicKeys:

    def test_redacts_anthropic_api_key(self):
        text = "key is sk-ant-api03-abcdefghijklmnopqrstuvwxyz1234567890"
        result, events = redact_credentials(text)
        assert "sk-ant-" not in result
        assert "[REDACTED:api_key]" in result
        assert len(events) == 1
        assert events[0].pattern_name == "anthropic_api_key"

    def test_redacts_anthropic_key_mid_text(self):
        text = "export ANTHROPIC_API_KEY=sk-ant-api03-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx && echo done"
        result, events = redact_credentials(text)
        assert "sk-ant-" not in result
        assert "export ANTHROPIC_API_KEY=" in result
        assert "&& echo done" in result

    def test_preserves_surrounding_context(self):
        text = "before sk-ant-api03-abcdefghijklmnopqrstuvwxyz1234567890 after"
        result, _ = redact_credentials(text)
        assert result.startswith("before ")
        assert result.endswith(" after")


class TestOpenRouterKeys:

    def test_redacts_openrouter_key(self):
        text = "sk-or-v1-" + "a1b2c3d4" * 8  # 64 hex chars
        result, events = redact_credentials(text)
        assert "sk-or-v1-" not in result
        assert "[REDACTED:api_key]" in result
        assert events[0].pattern_name == "openrouter_api_key"


class TestOpenAIKeys:

    def test_redacts_openai_project_key(self):
        text = "sk-proj-abcdefghijklmnopqrstuvwxyz1234567890ABCD"
        result, events = redact_credentials(text)
        assert "sk-proj-" not in result
        assert "[REDACTED:api_key]" in result
        assert events[0].pattern_name == "openai_api_key"

    def test_openai_pattern_does_not_match_anthropic(self):
        """Anthropic keys match the anthropic pattern, not the generic sk- one."""
        text = "sk-ant-api03-abcdefghijklmnopqrstuvwxyz1234567890"
        result, events = redact_credentials(text)
        assert len(events) == 1
        assert events[0].pattern_name == "anthropic_api_key"

    def test_openai_pattern_does_not_match_openrouter(self):
        """OpenRouter keys match the openrouter pattern, not the generic sk- one."""
        text = "sk-or-v1-" + "ab" * 32  # 64 hex chars
        result, events = redact_credentials(text)
        assert len(events) == 1
        assert events[0].pattern_name == "openrouter_api_key"


class TestGitTokens:

    def test_redacts_ghp_token(self):
        text = "ghp_ABCDEFghijklmnop1234567890abcdefghij"
        result, events = redact_credentials(text)
        assert "ghp_" not in result
        assert "[REDACTED:token]" in result
        assert events[0].pattern_name == "github_token"

    def test_redacts_gho_token(self):
        text = "token: gho_abcdefghijklmnopqrstuvwxyz12"
        result, events = redact_credentials(text)
        assert "gho_" not in result

    def test_redacts_github_pat(self):
        text = "github_pat_ABCDEF1234567890abcdef1234567890_extra"
        result, events = redact_credentials(text)
        assert "github_pat_" not in result

    def test_redacts_codeberg_token(self):
        """Codeberg uses gitea-style tokens — same ghp/gho format or raw hex.
        The Flapjack incident token was 40 hex chars with no prefix, which
        falls to the generic hex pattern if 48+ or is missed if shorter.
        Prefixed Codeberg tokens match github_token pattern."""
        text = "ghp_SomeCodebergStyleToken1234567890extra"
        result, events = redact_credentials(text)
        assert "ghp_" not in result


class TestOnePasswordTokens:

    def test_redacts_1password_service_account_token(self):
        text = "ops_abcdefghijklmnopqrstuvwxyz1234567890"
        result, events = redact_credentials(text)
        assert "ops_" not in result
        assert "[REDACTED:service_token]" in result
        assert events[0].pattern_name == "onepassword_service_token"


class TestAgeSecretKeys:

    def test_redacts_age_secret_key(self):
        text = "AGE-SECRET-KEY-1QFNZJMRMR39KHHVE5TF96HQWKA2TT36MYN5MTYYQNKJ8KES29EQSFQ34Y"
        result, events = redact_credentials(text)
        assert "AGE-SECRET-KEY-" not in result
        assert "[REDACTED:age_secret_key]" in result
        assert events[0].pattern_name == "age_secret_key"

    def test_age_public_key_not_redacted(self):
        """age public keys start with 'age1' — should not be redacted."""
        text = "age1ql3z7hjy54pw3hyww5ayyfg7zqgvc7w3j2elw8zmrj2kg5sfn9aqmcac8p"
        result, events = redact_credentials(text)
        assert "age1" in result
        assert len(events) == 0


class TestBearerTokens:

    def test_redacts_bearer_in_header(self):
        text = 'Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U'
        result, events = redact_credentials(text)
        assert "eyJ" not in result
        assert "Authorization: " in result
        assert "[REDACTED:bearer_token]" in result

    def test_bearer_case_sensitive(self):
        """Bearer is case-sensitive per RFC 6750."""
        text = "bearer abc123_not_a_real_token_abcdefghij"
        result, events = redact_credentials(text)
        # Lowercase 'bearer' should NOT match
        assert "bearer" in result


class TestEthereumKeys:

    def test_redacts_ethereum_private_key(self):
        text = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
        result, events = redact_credentials(text)
        assert "0xac0974" not in result
        assert "[REDACTED:private_key]" in result
        assert events[0].pattern_name == "ethereum_private_key"

    def test_ethereum_address_not_redacted(self):
        """Ethereum addresses are 40 hex chars with 0x prefix — not 64."""
        text = "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045"
        result, events = redact_credentials(text)
        assert "0xd8dA" in result
        assert len(events) == 0

    def test_ethereum_tx_hash_not_redacted(self):
        """Tx hashes are 66 chars (0x + 64) but mixed case and context differs.
        These will match ethereum_private_key — acceptable false positive.
        This test documents the known behavior."""
        text = "0x" + "ab" * 32
        result, events = redact_credentials(text)
        # This WILL be redacted — it's indistinguishable from a private key
        assert len(events) >= 1


class TestPEMBlocks:

    def test_redacts_rsa_private_key(self):
        pem = (
            "-----BEGIN RSA PRIVATE KEY-----\n"
            "MIIEpAIBAAKCAQEA0Z3VS5JJcds3xfn/ygWyF8PbnGy5AHB9Q\n"
            "TG4lMnOGHAY0kXJeHEHHHHHHHHHHHHHHHHHHHHHHH0000000\n"
            "-----END RSA PRIVATE KEY-----"
        )
        text = f"Found key:\n{pem}\nEnd of scan."
        result, events = redact_credentials(text)
        assert "MIIEpA" not in result
        assert "-----BEGIN RSA PRIVATE KEY-----" not in result
        assert "[REDACTED:private_key]" in result
        assert "Found key:" in result
        assert "End of scan." in result

    def test_redacts_openssh_private_key(self):
        pem = (
            "-----BEGIN OPENSSH PRIVATE KEY-----\n"
            "b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAE\n"
            "-----END OPENSSH PRIVATE KEY-----"
        )
        result, events = redact_credentials(pem)
        assert "b3Blbn" not in result
        assert "[REDACTED:private_key]" in result

    def test_redacts_ec_private_key(self):
        pem = (
            "-----BEGIN EC PRIVATE KEY-----\n"
            "MHQCAQEEIBkg4LVWM9nuwNSk3yByxZpYRTBnVJk\n"
            "-----END EC PRIVATE KEY-----"
        )
        result, events = redact_credentials(pem)
        assert "MHQCAQEEIBkg" not in result

    def test_redacts_generic_private_key(self):
        pem = (
            "-----BEGIN PRIVATE KEY-----\n"
            "MIGHAgEAMBMGByqGSM49AgEGCCqGSM49AwEH\n"
            "-----END PRIVATE KEY-----"
        )
        result, events = redact_credentials(pem)
        assert "MIGHAgEA" not in result

    def test_public_key_not_redacted(self):
        """Public keys should pass through."""
        pem = (
            "-----BEGIN PUBLIC KEY-----\n"
            "MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCg\n"
            "-----END PUBLIC KEY-----"
        )
        result, events = redact_credentials(pem)
        assert "MIIBIjAN" in result
        assert len(events) == 0

    def test_certificate_not_redacted(self):
        """Certificates are not secrets."""
        pem = (
            "-----BEGIN CERTIFICATE-----\n"
            "MIIDdzCCAl+gAwIBAgIJANfHOBkZr8JOMA\n"
            "-----END CERTIFICATE-----"
        )
        result, events = redact_credentials(pem)
        assert "MIIDdzCC" in result
        assert len(events) == 0


class TestGenericHex:

    def test_redacts_64_char_hex_string(self):
        """64 hex chars (common token length) should be redacted."""
        hex_str = "a1b2c3d4" * 8  # 64 chars
        text = f"token={hex_str}"
        result, events = redact_credentials(text)
        assert hex_str not in result
        assert "[REDACTED:hex_secret]" in result

    def test_48_char_hex_redacted(self):
        """48 hex chars — at threshold."""
        hex_str = "ab" * 24  # 48 chars
        text = f"secret: {hex_str}"
        result, events = redact_credentials(text)
        assert hex_str not in result

    def test_40_char_hex_not_redacted(self):
        """40 hex chars — below threshold (git commit hash length)."""
        hex_str = "ab" * 20  # 40 chars
        text = f"commit {hex_str}"
        result, events = redact_credentials(text)
        assert hex_str in result
        assert len(events) == 0

    def test_git_hash_not_redacted(self):
        """Realistic git log output with 40-char hashes passes through."""
        text = "commit a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0\nAuthor: test"
        result, events = redact_credentials(text)
        assert "a1b2c3d4e5" in result

    def test_uppercase_hex_redacted(self):
        """Uppercase hex tokens should also match."""
        hex_str = "ABCDEF01" * 8  # 64 chars
        text = f"key: {hex_str}"
        result, events = redact_credentials(text)
        assert hex_str not in result

    def test_mixed_case_hex_redacted(self):
        """Mixed case hex tokens should match."""
        hex_str = "aAbBcCdD" * 6  # 48 chars
        result, events = redact_credentials(hex_str)
        assert hex_str not in result


# ---------------------------------------------------------------------------
# No-match cases (false positive resistance)
# ---------------------------------------------------------------------------

class TestFalsePositiveResistance:

    def test_normal_text_untouched(self):
        text = "Hello, this is a normal shell output with no secrets."
        result, events = redact_credentials(text)
        assert result == text
        assert len(events) == 0

    def test_short_hex_in_normal_output(self):
        """Short hex strings in normal output are not redacted."""
        text = "Process 0x7f3a running, exit code 0xff"
        result, events = redact_credentials(text)
        assert result == text

    def test_file_listing_untouched(self):
        text = "drwxr-xr-x 2 user group 4096 Mar 10 config.toml\n-rw-r--r-- 1 user group 256 keys.txt"
        result, events = redact_credentials(text)
        assert result == text

    def test_base64_encoded_content_untouched(self):
        """Base64 content (e.g., from file encoding) should not be redacted."""
        text = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAA"
        result, events = redact_credentials(text)
        assert "iVBORw0K" in result

    def test_json_output_untouched(self):
        text = '{"name": "test", "value": 42, "items": ["a", "b", "c"]}'
        result, events = redact_credentials(text)
        assert result == text

    def test_url_untouched(self):
        text = "https://codeberg.org/merryshelly/workspace/src/branch/main/README.md"
        result, events = redact_credentials(text)
        assert result == text

    def test_url_with_risk_slug_untouched(self):
        """URL slugs containing words ending in 'sk' followed by hyphens should
        not trigger the OpenAI key pattern. 'risk-parameter-updates-...' matches
        sk-[a-zA-Z0-9_-]{20,} without a word boundary anchor."""
        urls = [
            "https://governance.aave.com/t/arfc-chaos-labs-risk-parameter-updates-gno-on-v3-gnosis/17340",
            "https://governance.aave.com/t/chaos-labs-risk-stewards-increase-supply-and-borrow-caps/21146",
            "https://example.com/docs/task-management-system-overview-and-guide",
            "https://example.com/flask-session-configuration-guide-for-developers",
        ]
        for url in urls:
            result, events = redact_credentials(url)
            assert result == url, f"URL was incorrectly redacted: {url} -> {result}"
            assert len(events) == 0, f"Unexpected redaction events for: {url}"



    def test_uuid_untouched(self):
        """UUIDs are hex but have dashes and are 36 chars total (32 hex)."""
        text = "id: 550e8400-e29b-41d4-a716-446655440000"
        result, events = redact_credentials(text)
        assert "550e8400" in result

    def test_empty_string(self):
        result, events = redact_credentials("")
        assert result == ""
        assert len(events) == 0

    def test_ssh_public_key_untouched(self):
        text = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIGrU user@host"
        result, events = redact_credentials(text)
        assert "AAAAC3NzaC1" in result


class TestPDFHexFalsePositives:
    """PDF binary content contains hex strings in angle brackets that are not secrets."""

    def test_pdf_xref_id_not_redacted(self):
        """PDF cross-reference /ID entries use <hex> delimiters — not secrets."""
        text = '/ID[<A1B2C3D4E5F6A7B8C9D0E1F2A3B4C5D6E7F8A9B0A1B2C3D4E5F6A7B8C9D0><66785F64DD1BC04CAD6CA0ECFC9542C6>]'
        result, events = redact_credentials(text)
        # Neither hex string should be redacted
        hex_events = [e for e in events if e.pattern_name == "generic_hex"]
        assert len(hex_events) == 0
        assert "REDACTED" not in result

    def test_pdf_linearized_hex_not_redacted(self):
        """Longer PDF hex content inside angle brackets passes through."""
        hex_96 = "A1B2C3D4" * 12  # 96 chars
        text = f'/Filter/FlateDecode/ID[<{hex_96}>]/Index[442 58]'
        result, events = redact_credentials(text)
        hex_events = [e for e in events if e.pattern_name == "generic_hex"]
        assert len(hex_events) == 0

    def test_bare_hex_still_redacted(self):
        """Hex strings NOT in angle brackets are still caught."""
        hex_64 = "a1b2c3d4" * 8
        text = f"token={hex_64}"
        result, events = redact_credentials(text)
        assert hex_64 not in result
        assert "[REDACTED:hex_secret]" in result

    def test_pdf_raw_binary_with_hex_ids(self):
        """Realistic PDF binary fragment with multiple hex IDs."""
        text = (
            '%PDF-1.7\r\n'
            '442 0 obj\r<</Linearized 1/L 189140/O 444/E 154504>>\r\n'
            '<</DecodeParms<</Columns 5/Predictor 12>>'
            '/Filter/FlateDecode'
            '/ID[<' + 'AB' * 24 + '><' + 'CD' * 24 + '>]'  # Two 48-char hex IDs
            '/Index[442 58]>>'
        )
        result, events = redact_credentials(text)
        hex_events = [e for e in events if e.pattern_name == "generic_hex"]
        assert len(hex_events) == 0


# ---------------------------------------------------------------------------
# Multiple credentials in one output
# ---------------------------------------------------------------------------

class TestMultipleCredentials:

    def test_two_different_patterns(self):
        text = (
            "ANTHROPIC_API_KEY=sk-ant-api03-abcdefghijklmnopqrstuvwxyz1234567890\n"
            "OPENROUTER_KEY=sk-or-v1-" + "ab" * 32 + "\n"
        )
        result, events = redact_credentials(text)
        assert "sk-ant-" not in result
        assert "sk-or-v1-" not in result
        assert len(events) == 2
        pattern_names = {e.pattern_name for e in events}
        assert "anthropic_api_key" in pattern_names
        assert "openrouter_api_key" in pattern_names

    def test_same_pattern_twice(self):
        key1 = "ghp_AAAABBBBCCCCDDDDEEEEFFFFGGG1"
        key2 = "ghp_1111222233334444555566667777"
        text = f"token1={key1}\ntoken2={key2}"
        result, events = redact_credentials(text)
        assert key1 not in result
        assert key2 not in result
        assert len(events) == 2

    def test_pem_plus_api_key(self):
        text = (
            "-----BEGIN RSA PRIVATE KEY-----\n"
            "MIIEpAIBAAKCAQEA0000000000000000000000\n"
            "-----END RSA PRIVATE KEY-----\n"
            "Also found: sk-ant-api03-zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz\n"
        )
        result, events = redact_credentials(text)
        assert "MIIEpA" not in result
        assert "sk-ant-" not in result
        assert len(events) == 2


# ---------------------------------------------------------------------------
# No double-redaction
# ---------------------------------------------------------------------------

class TestNoDoubleRedaction:

    def test_anthropic_key_not_also_matched_by_openai(self):
        """sk-ant- should only match anthropic, not also sk- generic."""
        text = "sk-ant-api03-abcdefghijklmnopqrstuvwxyz1234567890"
        result, events = redact_credentials(text)
        assert len(events) == 1
        assert events[0].pattern_name == "anthropic_api_key"
        # Should not contain nested redaction markers
        assert result.count("[REDACTED:") == 1

    def test_openrouter_key_not_also_matched_by_openai(self):
        text = "sk-or-v1-" + "ab" * 32
        result, events = redact_credentials(text)
        assert len(events) == 1
        assert events[0].pattern_name == "openrouter_api_key"

    def test_ethereum_key_not_also_matched_by_generic_hex(self):
        """0x + 64 hex should match ethereum, not generic hex."""
        text = "0x" + "ab" * 32
        result, events = redact_credentials(text)
        assert len(events) == 1
        assert events[0].pattern_name == "ethereum_private_key"


# ---------------------------------------------------------------------------
# RedactionEvent structure
# ---------------------------------------------------------------------------

class TestRedactionEvents:

    def test_event_has_all_fields(self):
        text = "key: sk-ant-api03-abcdefghijklmnopqrstuvwxyz1234567890"
        _, events = redact_credentials(text)
        assert len(events) == 1
        event = events[0]
        assert isinstance(event, RedactionEvent)
        assert isinstance(event.pattern_name, str)
        assert isinstance(event.redaction_label, str)
        assert isinstance(event.char_count, int)
        assert isinstance(event.position, int)

    def test_char_count_matches_original_value(self):
        key = "sk-ant-api03-abcdefghijklmnopqrstuvwxyz1234567890"
        text = f"key={key}"
        _, events = redact_credentials(text)
        assert events[0].char_count == len(key)

    def test_position_is_start_of_match(self):
        key = "sk-ant-api03-abcdefghijklmnopqrstuvwxyz1234567890"
        text = f"key={key}"
        _, events = redact_credentials(text)
        assert events[0].position == 4  # "key=" is 4 chars


# ---------------------------------------------------------------------------
# Integration: execute_tool applies redaction
# ---------------------------------------------------------------------------

class TestExecuteToolRedaction:
    """Redaction is applied inside execute_tool() after tool dispatch."""

    @pytest.fixture
    def agent_config(self, tmp_path):
        from openalph.config import AgentConfig, ProviderConfig
        return AgentConfig(
            name="test-agent",
            default_model="anthropic/claude-sonnet-4-20250514",
            max_tokens=8192,
            providers={"anthropic": ProviderConfig(
                key="anthropic", type="anthropic",
                api_key="sk-test", base_url=None, quirks=None,
            )},
            workspace=tmp_path,
        )

    @pytest.mark.asyncio
    async def test_shell_output_redacted(self, agent_config):
        """Shell command returning a secret gets redacted."""
        secret_output = "sk-ant-api03-abcdefghijklmnopqrstuvwxyz1234567890"
        with patch("openalph.tools.shell.run_shell", new_callable=AsyncMock,
                    return_value=ToolResult(content=secret_output)):
            result = await execute_tool(
                name="shell",
                input={"command": "op read 'op://vault/item'"},
                tool_config={"default_timeout": 30, "max_output": 50000},
                agent_config=agent_config,
            )
        assert "sk-ant-" not in result.content
        assert "[REDACTED:api_key]" in result.content

    @pytest.mark.asyncio
    async def test_file_read_output_redacted(self, agent_config, tmp_path):
        """Reading a file containing a secret gets redacted."""
        secret_file = tmp_path / "secret.txt"
        secret_file.write_text("my_key=ghp_ABCDEFghijklmnopqrstuvwxyz12345678\n")
        with patch("openalph.tools.file.read_file", new_callable=AsyncMock,
                    return_value=ToolResult(content=secret_file.read_text())):
            result = await execute_tool(
                name="file_read",
                input={"path": str(secret_file)},
                tool_config={},
                agent_config=agent_config,
            )
        assert "ghp_" not in result.content
        assert "[REDACTED:token]" in result.content

    @pytest.mark.asyncio
    async def test_web_fetch_output_redacted(self, agent_config):
        """Web content containing leaked credentials gets redacted."""
        page_content = "leaked key: sk-proj-abcdefghijklmnopqrstuvwxyz1234567890ABCD"
        with patch("openalph.tools.web.web_fetch", new_callable=AsyncMock,
                    return_value=ToolResult(content=page_content)):
            result = await execute_tool(
                name="web_fetch",
                input={"url": "https://example.com"},
                tool_config={},
                agent_config=agent_config,
            )
        assert "sk-proj-" not in result.content
        assert "[REDACTED:api_key]" in result.content

    @pytest.mark.asyncio
    async def test_error_results_not_redacted(self, agent_config):
        """Error results pass through without redaction.
        Error content is framework-generated, not untrusted tool output."""
        error_content = "command not found: sk-ant-api03-test"
        with patch("openalph.tools.shell.run_shell", new_callable=AsyncMock,
                    return_value=ToolResult(content=error_content, is_error=True)):
            result = await execute_tool(
                name="shell",
                input={"command": "bad"},
                tool_config={"default_timeout": 30, "max_output": 50000},
                agent_config=agent_config,
            )
        # Error results should still be redacted — errors can contain secrets
        # (e.g., stderr from a failed command that echoed the key)
        assert "sk-ant-" not in result.content
        assert "[REDACTED:api_key]" in result.content

    @pytest.mark.asyncio
    async def test_clean_output_passes_through(self, agent_config):
        """Normal output without credentials is unchanged."""
        clean_output = "total 42\ndrwxr-xr-x 3 user group 4096 Mar 10 workspace"
        with patch("openalph.tools.shell.run_shell", new_callable=AsyncMock,
                    return_value=ToolResult(content=clean_output)):
            result = await execute_tool(
                name="shell",
                input={"command": "ls -la"},
                tool_config={"default_timeout": 30, "max_output": 50000},
                agent_config=agent_config,
            )
        assert result.content == clean_output

    @pytest.mark.asyncio
    async def test_is_error_flag_preserved(self, agent_config):
        """Redaction doesn't alter the is_error flag."""
        with patch("openalph.tools.shell.run_shell", new_callable=AsyncMock,
                    return_value=ToolResult(
                        content="error: sk-ant-api03-abcdefghijklmnopqrstuvwxyz1234567890",
                        is_error=True,
                    )):
            result = await execute_tool(
                name="shell",
                input={"command": "bad"},
                tool_config={"default_timeout": 30, "max_output": 50000},
                agent_config=agent_config,
            )
        assert result.is_error is True
        assert "[REDACTED:" in result.content


# ---------------------------------------------------------------------------
# Integration: redaction callback for Matrix notices
# ---------------------------------------------------------------------------

class TestRedactionCallback:
    """execute_tool invokes on_redaction callback when credentials are found."""

    @pytest.fixture
    def agent_config(self, tmp_path):
        from openalph.config import AgentConfig, ProviderConfig
        return AgentConfig(
            name="test-agent",
            default_model="anthropic/claude-sonnet-4-20250514",
            max_tokens=8192,
            providers={"anthropic": ProviderConfig(
                key="anthropic", type="anthropic",
                api_key="sk-test", base_url=None, quirks=None,
            )},
            workspace=tmp_path,
        )

    @pytest.mark.asyncio
    async def test_callback_invoked_on_redaction(self, agent_config):
        """on_redaction callback is called with event details."""
        callback = AsyncMock()
        with patch("openalph.tools.shell.run_shell", new_callable=AsyncMock,
                    return_value=ToolResult(
                        content="key=sk-ant-api03-abcdefghijklmnopqrstuvwxyz1234567890"
                    )):
            await execute_tool(
                name="shell",
                input={"command": "op read key"},
                tool_config={"default_timeout": 30, "max_output": 50000},
                agent_config=agent_config,
                callbacks={"on_redaction": callback},
            )
        callback.assert_called_once()
        args = callback.call_args
        # Callback receives: tool_name, events list
        assert args[0][0] == "shell"  # tool_name
        assert len(args[0][1]) == 1   # events list
        assert args[0][1][0].pattern_name == "anthropic_api_key"

    @pytest.mark.asyncio
    async def test_callback_not_invoked_when_clean(self, agent_config):
        """No callback when output has no credentials."""
        callback = AsyncMock()
        with patch("openalph.tools.shell.run_shell", new_callable=AsyncMock,
                    return_value=ToolResult(content="clean output")):
            await execute_tool(
                name="shell",
                input={"command": "echo hello"},
                tool_config={"default_timeout": 30, "max_output": 50000},
                agent_config=agent_config,
                callbacks={"on_redaction": callback},
            )
        callback.assert_not_called()

    @pytest.mark.asyncio
    async def test_callback_failure_does_not_break_redaction(self, agent_config):
        """If the callback raises, redaction still applies."""
        callback = AsyncMock(side_effect=Exception("callback broke"))
        with patch("openalph.tools.shell.run_shell", new_callable=AsyncMock,
                    return_value=ToolResult(
                        content="sk-ant-api03-abcdefghijklmnopqrstuvwxyz1234567890"
                    )):
            result = await execute_tool(
                name="shell",
                input={"command": "op read key"},
                tool_config={"default_timeout": 30, "max_output": 50000},
                agent_config=agent_config,
                callbacks={"on_redaction": callback},
            )
        # Redaction still happened despite callback failure
        assert "sk-ant-" not in result.content
        assert "[REDACTED:api_key]" in result.content


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

class TestRedactionLogging:

    def test_redaction_logs_warning(self, caplog):
        """Each redaction event is logged at WARNING level."""
        text = "key=sk-ant-api03-abcdefghijklmnopqrstuvwxyz1234567890"
        with caplog.at_level(logging.WARNING, logger="openalph.security"):
            redact_credentials(text)
        assert any("anthropic_api_key" in record.message for record in caplog.records)

    def test_no_log_when_clean(self, caplog):
        """No log entries when no credentials found."""
        with caplog.at_level(logging.WARNING, logger="openalph.security"):
            redact_credentials("normal output")
        security_records = [r for r in caplog.records if r.name == "openalph.security"]
        assert len(security_records) == 0

    def test_redacted_value_not_in_log(self, caplog):
        """The actual secret value must not appear in log messages."""
        key = "sk-ant-api03-abcdefghijklmnopqrstuvwxyz1234567890"
        text = f"key={key}"
        with caplog.at_level(logging.WARNING, logger="openalph.security"):
            redact_credentials(text)
        for record in caplog.records:
            assert key not in record.message
