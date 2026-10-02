from vector_embed.core.privacy.mask import PersonalRedactor, redact_personal

RESUME = """Jane Q. Doe
jane.doe@example.com | +1 415 555 0132
https://www.linkedin.com/in/janedoe | github.com/janedoe
42 Market Street, San Francisco, USA

Senior engineer at Acme (2019-2024). Reach Jane Q. Doe at jane.doe@example.com.
"""


def test_personal_details_become_placeholders() -> None:
    result = redact_personal(RESUME)
    text = result.text
    for original in (
        "jane.doe@example.com",
        "415 555 0132",
        "linkedin.com/in/janedoe",
        "github.com/janedoe",
        "42 Market Street",
        "Jane Q. Doe",
    ):
        assert original not in text
    assert "[EMAIL_1]" in text
    assert "[PHONE_1]" in text
    assert "[NAME_1]" in text
    assert "[ADDRESS_1]" in text
    assert "[URL_1]" in text
    assert "[URL_2]" in text


def test_city_country_dates_and_ranges_are_kept() -> None:
    text = redact_personal(RESUME).text
    assert "San Francisco, USA" in text
    assert "2019-2024" in text
    assert "Senior engineer at Acme" in text


def test_the_same_value_gets_the_same_placeholder_and_restore_roundtrips() -> None:
    result = redact_personal(RESUME)
    assert result.text.count("[EMAIL_1]") == 2
    assert result.count == len(result.mapping)
    assert result.restore(result.text) == RESUME
    answer = "Contact [NAME_1] at [EMAIL_1] or [PHONE_1]."
    assert (
        result.restore(answer) == "Contact Jane Q. Doe at jane.doe@example.com or +1 415 555 0132."
    )


def test_known_names_and_labelled_names_are_redacted() -> None:
    result = redact_personal(
        "Name: Rajesh Kumar\nAlso known as Raj K. Reference: Priya Nair", ["Priya Nair"]
    )
    assert "Rajesh Kumar" not in result.text
    assert "Priya Nair" not in result.text
    assert result.text.count("[NAME_") == 2


def test_blank_known_names_are_ignored() -> None:
    assert redact_personal("hello world", ["", "  "]).text == "hello world"


def test_nothing_to_redact() -> None:
    result = redact_personal("Built payment systems in Python for 2M users.")
    assert result.text == "Built payment systems in Python for 2M users."
    assert result.mapping == {}


def test_short_digit_runs_are_not_phone_numbers() -> None:
    result = redact_personal("Served 2000-2010 and 12345 users, order 1234-5678")
    assert result.mapping == {}


def test_numbering_is_per_redactor_instance() -> None:
    first = PersonalRedactor().redact("a@x.com")
    second = PersonalRedactor().redact("b@y.com")
    assert first.text == second.text == "[EMAIL_1]"
    assert first.mapping != second.mapping


def test_names_are_matched_as_whole_words_in_any_case() -> None:
    redactor = PersonalRedactor(["Ann Lee"])
    result = redactor.redact("ANN LEE applied. Annual review for Ann Lee and ann lee.")
    assert "Annual review" in result.text  # "Ann" inside a longer word is not a name
    assert "ANN LEE" not in result.text
    assert "ann lee" not in result.text
    assert result.text.count("[NAME_1]") == 3
    assert result.restore("[NAME_1]") == "Ann Lee"


def test_all_caps_heading_is_a_name() -> None:
    result = redact_personal("JANE DOE\nPython developer\nWork Experience\nAcme")
    assert "JANE DOE" not in result.text
    assert "Python developer" in result.text


def test_job_title_lines_are_not_names() -> None:
    for title in ("Senior Software Engineer", "Product Manager", "DATA SCIENTIST"):
        assert redact_personal(f"{title}\nBuilt things").text.startswith(title)


def test_a_name_found_later_still_masks_an_earlier_occurrence() -> None:
    redactor = PersonalRedactor()
    redactor.learn("Name: Priya Nair\nrest")
    assert "Priya Nair" not in redactor.redact("Earlier message mentioning Priya Nair.").text


def test_unused_known_names_do_not_consume_placeholder_numbers() -> None:
    result = redact_personal("Email jane@example.com", ["Nobody Here"])
    assert result.mapping == {"[EMAIL_1]": "jane@example.com"}
