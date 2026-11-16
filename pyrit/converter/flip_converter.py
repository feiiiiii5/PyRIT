# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pyrit.common.text_helper import graphemes
from pyrit.converter.converter import Converter, ConverterResult
from pyrit.models import PromptDataType


class FlipConverter(Converter):
    """
    Flips the input text prompt. For example, "hello me" would be converted to "em olleh".

    Reversal operates on extended grapheme clusters, so emoji ZWJ sequences, flags,
    skin-tone modifiers, keycaps and combining marks stay intact as user-perceived
    characters. Because the reversed text is segmented again when it is read, the
    operation is not always an involution: adjacent regional indicators can pair up
    differently, and a combining mark at the start of the input ends up after the
    first character of the output, so ``flip(flip(x)) != x`` for those inputs.

    Indic conjunct clustering follows the ``regex`` module's Unicode tables, so the
    exact clusters can vary with that dependency.
    """

    SUPPORTED_INPUT_TYPES = ("text",)
    SUPPORTED_OUTPUT_TYPES = ("text",)

    async def convert_async(self, *, prompt: str, input_type: PromptDataType = "text") -> ConverterResult:
        """
        Convert the given prompt by reversing the text.

        Args:
            prompt: The prompt to be converted.
            input_type: Type of data.

        Returns:
            The converted text representation of the original prompt with characters reversed.

        Raises:
            ValueError: If the input type is not supported.
        """
        if not self.input_supported(input_type):
            raise ValueError("Input type not supported")

        rev_prompt = "".join(reversed(graphemes(prompt)))

        return ConverterResult(output_text=rev_prompt, output_type="text")
