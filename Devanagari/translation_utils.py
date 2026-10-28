from google import genai

GEMINI_API_KEY = "AQ.Ab8RN6Kwg2y-SUbWYMm0pwkxQcMWK13Q8GWdFviqxz8Q_ezA3w"

client = genai.Client(api_key=GEMINI_API_KEY)


def translate_sanskrit_to_kannada(sanskrit_text):

    if not sanskrit_text or not sanskrit_text.strip():
        return "Translation error: No Sanskrit text provided."

    try:

        lines = [
            " ".join(line.strip().split())
            for line in sanskrit_text.splitlines()
            if line.strip()
        ]

        cleaned_text = "\n".join(lines)

        system_instruction = """
You are the Sanskrit-to-Kannada translation engine
of the NeoLipi ancient inscription recognition system.

The input is Sanskrit text obtained from OCR of an
ancient Devanagari/Sanskrit stone inscription.

Translate the Sanskrit inscription into meaningful
modern Kannada.

Rules:

1. Understand the complete sentence before translating.
2. Preserve the original meaning.
3. Consider Sanskrit grammar and sentence structure.
4. Consider historical inscriptional context.
5. Preserve names of people and places.
6. Preserve historical terminology.
7. Correct obvious OCR mistakes only when the intended
   Sanskrit word is reasonably clear.
8. Do not invent information.
9. Do not provide alternative translations.
10. Do not provide explanations.
11. Do not provide analysis.
12. Do not provide transliteration.
13. Do not provide English.
14. Return ONLY the final Kannada translation.
"""

        prompt = f"""
Translate the following Sanskrit OCR text into modern Kannada.

Preserve the original meaning and historical context.

Return ONLY Kannada.

SANSKRIT OCR TEXT:

{cleaned_text}
"""

        interaction = client.interactions.create(
            model="gemini-3.8-flash",
            system_instruction=system_instruction,
            input=prompt,
           generation_config={
    "seed": 12345,
    "thinking_level": "low",
    "max_output_tokens": 300
}
        )

        result = interaction.output_text.strip()

        if not result:
            return "Translation error: Empty response from Gemini."

        return result

    except Exception as e:
        return f"Translation error: {e}"