import pytest

from vector_embed.core.privacy import detectors as d
from vector_embed.core.privacy.detectors import detect_sensitive
from vector_embed.core.privacy.mask import mask_sensitive

VALID_CARD = "4111 1111 1111 1111"  # Visa test number, Luhn valid
VALID_AADHAAR = "2345 6789 0124"  # Verhoeff valid (computed below in a test)
VALID_IBAN = "GB82 WEST 1234 5698 7654 32"


def kinds(text: str) -> list[str]:
    return [f.kind for f in detect_sensitive(text)]


class TestChecksums:
    def test_luhn(self) -> None:
        assert d.luhn_valid("4111111111111111")
        assert not d.luhn_valid("4111111111111112")
        assert not d.luhn_valid("0")

    def test_verhoeff_roundtrip(self) -> None:
        base = "23456789012"
        check = next(str(c) for c in range(10) if d.verhoeff_valid(base + str(c)))
        assert d.verhoeff_valid(base + check)
        wrong = str((int(check) + 1) % 10)
        assert not d.verhoeff_valid(base + wrong)

    def test_iban(self) -> None:
        assert d.iban_valid(VALID_IBAN)
        assert not d.iban_valid("GB82 WEST 1234 5698 7654 33")
        assert not d.iban_valid("GB82")

    def test_aba_routing(self) -> None:
        assert d.aba_valid("021000021")
        assert not d.aba_valid("021000022")
        assert not d.aba_valid("1234")


class TestDetection:
    def test_ssn_with_dashes_and_invalid_areas(self) -> None:
        assert kinds("SSN 123-45-6789") == ["ssn"]
        assert kinds("fake 000-12-3456 666-12-3456 900-12-3456 123-00-4567 123-45-0000") == []

    def test_unformatted_ssn_needs_a_label(self) -> None:
        assert kinds("Social Security: 123456789") == ["ssn"]
        assert kinds("order id 123456789") == []

    def test_credit_cards_are_luhn_validated(self) -> None:
        assert kinds(f"Card {VALID_CARD} exp 12/27") == ["card"]
        assert kinds("Card 4111-1111-1111-1111") == ["card"]
        assert kinds("number 4111 1111 1111 1112") == []
        assert kinds("0000 0000 0000 0000") == []

    def test_aadhaar_needs_a_valid_checksum(self) -> None:
        base = "23456789012"
        check = next(str(c) for c in range(10) if d.verhoeff_valid(base + str(c)))
        number = base + check
        spaced = f"{number[:4]} {number[4:8]} {number[8:]}"
        assert kinds(f"Aadhaar {spaced}") == ["aadhaar"]
        wrong = base + str((int(check) + 1) % 10)
        assert kinds(f"id {wrong}") == []

    def test_pan_and_uk_ni_and_canadian_sin(self) -> None:
        assert kinds("PAN ABCPE1234F") == ["pan"]
        assert kinds("PAN ABCDE1234F") == []  # fourth letter must be a holder type
        assert kinds("NI number AB 12 34 56 C") == ["ni_number"]
        assert kinds("SIN 046 454 286") == ["sin"]
        assert kinds("046 454 286") == []  # no label, no match

    def test_labelled_passport_and_drivers_licence(self) -> None:
        assert kinds("Passport No: K1234567") == ["passport"]
        assert kinds("passport number N9876543") == ["passport"]
        assert kinds("Driver's License: D123-4567-8901") == ["drivers_license"]
        assert kinds("DL# S1234567") == ["drivers_license"]
        assert kinds("passport: Renewal") == []
        assert kinds("K1234567 on its own") == []

    def test_tax_ids_accounts_routing_and_iban(self) -> None:
        assert kinds("EIN 12-3456789") == ["tax_id"]
        assert kinds("Tax ID: 123456789") == ["tax_id"]
        assert kinds("Account number: 000123456789") == ["bank_account"]
        assert kinds("Routing: 021000021") == ["routing"]
        assert kinds("Routing: 021000022") == []
        assert kinds(f"IBAN {VALID_IBAN}") == ["iban"]

    @pytest.mark.parametrize(
        "secret",
        [
            "sk-abcdefghijklmnopqrstuvwxyz123456",
            "AKIAIOSFODNN7EXAMPLE",
            "ghp_" + "a" * 36,
            "api_key = 'abcd1234efgh5678'",
            "password: hunter2hunter2",
            "-----BEGIN RSA PRIVATE KEY-----\nMIIE\n-----END RSA PRIVATE KEY-----",
            "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.abcdefghijklmnop",
        ],
    )
    def test_secrets(self, secret: str) -> None:
        assert kinds(f"config {secret} end") == ["secret"]

    @pytest.mark.parametrize(
        "text",
        [
            "Worked at Acme 2019-2024 and 2015-2019",
            "Call me on +1 415 555 0132 or (415) 555-0132",
            "GPA 3.85 / 4.0, graduated 2014",
            "Zip 94107, floor 12, room 4012",
            "Version 1.2.3 released 2024-05-17 build 20240517",
            "Served 2,000,000 users with 99.99% uptime",
            "ISBN 9780306406157 and order 1234567890123",
        ],
    )
    def test_ordinary_numbers_are_not_masked(self, text: str) -> None:
        assert detect_sensitive(text) == []

    def test_longest_overlap_wins_and_results_are_ordered(self) -> None:
        text = f"ssn 123-45-6789 then card {VALID_CARD} and key sk-abcdefghijklmnopqrstuvwxyz"
        found = detect_sensitive(text)
        assert [f.kind for f in found] == ["ssn", "card", "secret"]
        assert [f.start for f in found] == sorted(f.start for f in found)

    def test_a_tax_id_that_looks_like_an_ssn_is_reported_once(self) -> None:
        assert len(detect_sensitive("Tax ID 123-45-6789")) == 1


class TestMasking:
    def test_ids_are_replaced_irreversibly(self) -> None:
        text = f"Jane, SSN 123-45-6789, card {VALID_CARD}, passport no K1234567, emp 2019-2024"
        result = mask_sensitive(text)
        assert result.text == (
            "Jane, SSN [SSN REMOVED], card [CARD REMOVED], passport no [PASSPORT REMOVED], "
            "emp 2019-2024"
        )
        assert "123-45-6789" not in result.text
        assert [f.kind for f in result.findings] == ["ssn", "card", "passport"]

    def test_clean_text_is_unchanged(self) -> None:
        result = mask_sensitive("Senior engineer, 8 years, Python and SQL.")
        assert result.text == "Senior engineer, 8 years, Python and SQL."
        assert result.findings == []

    def test_every_fixture_id_family_is_removed(self) -> None:
        base = "23456789012"
        check = next(str(c) for c in range(10) if d.verhoeff_valid(base + str(c)))
        resume = "\n".join(
            [
                "SSN: 123-45-6789",
                "Passport No: K1234567",
                "Driver's License: D123-4567-8901",
                f"Aadhaar: {base + check}",
                "PAN: ABCPE1234F",
                f"Card: {VALID_CARD}",
                "Routing: 021000021",
            ]
        )
        masked = mask_sensitive(resume).text
        for secret in (
            "123-45-6789",
            "K1234567",
            "D123-4567-8901",
            base + check,
            "ABCPE1234F",
            "4111",
            "021000021",
        ):
            assert secret not in masked
        assert masked.count("REMOVED") == 7
