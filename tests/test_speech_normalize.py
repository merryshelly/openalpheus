"""Speech text normalizer tests (im7t.68) — parity with scripts/test_tts_say.py.

The normalizer is a verbatim port of scripts/tts-say's normalize(): the same five
passes in the same order. These tests are the parity contract — the port must not
"improve" any rule, and must keep the byte-identical passthrough guarantee for text
no rule matches.

Spec: memory/projects/openalph/specs/im7t.68-tts-stt-native-tools-spec.md
"""

import pytest

from openalph.tools.speech import normalize


def check(src, expected):
    actual = normalize(src)
    assert actual == expected, f"input: {src!r}\nexpected: {expected!r}\nactual:   {actual!r}"


# --- Rule 1-2: version numbers (3+ dotted parts) -----------------------------


def test_version_0_19_0():
    check("Version 0.19.0 shipped", "Version zero point nineteen point zero shipped")


def test_version_after_product_name():
    check("mlx-audio 0.4.2 is running", "mlx-audio zero point four point two is running")


# --- Rule 3: CVE identifiers -------------------------------------------------


def test_cve_identifier():
    check("CVE-2026-5281 affects it", "C V E twenty twenty-six fifty-two eighty-one affects it")


# --- Rule 4-5: IPv4 addresses ------------------------------------------------


def test_ip_with_host_word():
    check("host 10.0.20.104 is up", "host ten dot zero dot twenty dot one oh four is up")


def test_bare_ip():
    check("10.0.20.101", "ten dot zero dot twenty dot one oh one")


# --- Rule 6-8: port numbers --------------------------------------------------


def test_port_8006():
    check("port 8006", "port eight thousand and six")


def test_port_8001():
    check("port 8001", "port eight thousand and one")


def test_port_8000():
    check("port 8000", "port eight thousand")


# --- Rule 9: RFC + 4-digit ---------------------------------------------------


def test_rfc_1918():
    check("RFC 1918", "R F C nineteen eighteen")


# --- Rule 10: allow-listed acronyms only -------------------------------------


def test_allowlisted_acronyms():
    check("HTTPS and SSH", "H T T P S and S S H")


def test_non_allowlisted_caps_untouched():
    check("KOKORO and MLX-AUDIO are fine", "KOKORO and MLX-AUDIO are fine")


# --- Rule 11: passthrough is byte-identical ----------------------------------


def test_plain_text_passthrough_byte_identical():
    s = "Hello there, this is a test."
    out = normalize(s)
    assert out == s
    assert out.encode("utf-8") == s.encode("utf-8")


# --- Rule 12: two-component decimals are NOT versions ------------------------


@pytest.mark.parametrize("s", ["2.5 dB", "3.5x speedup"])
def test_two_component_decimal_untouched(s):
    assert normalize(s) == s
    assert normalize(s).encode("utf-8") == s.encode("utf-8")


# --- Robustness / disambiguation ---------------------------------------------


def test_four_part_dotted_is_ip_not_version_tail():
    check("10.0.20.104", "ten dot zero dot twenty dot one oh four")


def test_ip_with_trailing_sentence_period():
    check("connect to 10.0.20.104.", "connect to ten dot zero dot twenty dot one oh four.")


def test_five_dotted_parts_read_as_version():
    check("v1.2.3.4.5", "vone point two point three point four point five")


def test_non_octet_first_part_is_version_not_ip_tail():
    check(
        "1111.2.3.4",
        "one thousand and one hundred and eleven point two point three point four",
    )


def test_valid_ip_11_2_3_4():
    check("11.2.3.4", "eleven dot two dot three dot four")


def test_mixed_sentence():
    check(
        "Version 0.19.0 and CVE-2026-5281 on 10.0.20.104",
        "Version zero point nineteen point zero and C V E twenty twenty-six "
        "fifty-two eighty-one on ten dot zero dot twenty dot one oh four",
    )


def test_port_variants():
    check("port 8080", "port eight thousand and eighty")
    check("port:443", "port:four hundred and forty-three")


def test_empty_string_passthrough():
    assert normalize("") == ""


def test_normalize_is_idempotent_on_output():
    """Normalized output contains no dotted/4-digit/CVE shapes to re-match."""
    once = normalize("CVE-2026-5281 on 10.0.20.104 port 8006 v0.19.0")
    assert normalize(once) == once
