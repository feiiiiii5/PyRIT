# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest

from pyrit.converter import ConverterResult, FlipConverter


async def test_flip_converter_basic():
    converter = FlipConverter()
    result = await converter.convert_async(prompt="hello", input_type="text")
    assert isinstance(result, ConverterResult)
    assert result.output_text == "olleh"
    assert result.output_type == "text"


async def test_flip_converter_sentence():
    converter = FlipConverter()
    result = await converter.convert_async(prompt="hello world", input_type="text")
    assert isinstance(result, ConverterResult)
    assert result.output_text == "dlrow olleh"
    assert result.output_type == "text"


async def test_flip_converter_empty():
    converter = FlipConverter()
    result = await converter.convert_async(prompt="", input_type="text")
    assert isinstance(result, ConverterResult)
    assert result.output_text == ""
    assert result.output_type == "text"


async def test_flip_converter_single_char():
    converter = FlipConverter()
    result = await converter.convert_async(prompt="a", input_type="text")
    assert isinstance(result, ConverterResult)
    assert result.output_text == "a"
    assert result.output_type == "text"


async def test_flip_converter_palindrome():
    converter = FlipConverter()
    result = await converter.convert_async(prompt="racecar", input_type="text")
    assert isinstance(result, ConverterResult)
    assert result.output_text == "racecar"
    assert result.output_type == "text"


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        ("🇮🇳 flag", "galf 🇮🇳"),
        ("👨‍👩‍👧 family", "ylimaf 👨‍👩‍👧"),
        ("👍🏽 OK", "KO 👍🏽"),
        ("1️⃣ test", "tset 1️⃣"),
        ("cafe\u0301", "e\u0301fac"),
        ("\u0928\u092e\u0938\u094d\u0924\u0947", "\u0938\u094d\u0924\u0947\u092e\u0928"),
        ("hello world", "dlrow olleh"),
    ],
)
async def test_flip_converter_reverses_grapheme_clusters_async(*, prompt: str, expected: str) -> None:
    result = await FlipConverter().convert_async(prompt=prompt)

    assert result.output_text == expected


async def test_flip_converter_input_not_supported():
    converter = FlipConverter()
    with pytest.raises(ValueError):
        await converter.convert_async(prompt="hello", input_type="image_path")


async def test_flip_converter_double_flip_is_identity_for_indic_conjuncts() -> None:
    converter = FlipConverter()
    once = await converter.convert_async(prompt="\u0915\u094d\u0937\u092e\u093e")
    twice = await converter.convert_async(prompt=once.output_text)

    assert twice.output_text == "\u0915\u094d\u0937\u092e\u093e"


@pytest.mark.parametrize(
    "prompt",
    [
        "\U0001f1ee\U0001f1f3\U0001f1fa\U0001f1f8\U0001f1eb",
        "\u0301abc",
    ],
)
async def test_flip_converter_is_not_involutive_when_reversed_clusters_resegment(prompt: str) -> None:
    converter = FlipConverter()
    once = await converter.convert_async(prompt=prompt)
    twice = await converter.convert_async(prompt=once.output_text)

    assert twice.output_text != prompt
