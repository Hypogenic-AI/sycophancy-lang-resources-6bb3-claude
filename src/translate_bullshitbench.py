"""Translate BullshitBench v2 questions into 6 non-English languages using GPT-4.1."""

import json
import os
import asyncio
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

LANGUAGES = {
    "fr": "French",
    "ar": "Arabic",
    "hi": "Hindi",
    "sw": "Swahili",
    "yo": "Yoruba",
    "tl": "Tagalog",
}

SEMAPHORE = asyncio.Semaphore(20)  # rate limit


async def translate_question(question_text: str, lang_code: str, lang_name: str) -> str:
    """Translate a single question to the target language."""
    async with SEMAPHORE:
        resp = await client.chat.completions.create(
            model="gpt-4.1",
            temperature=0.3,
            max_tokens=1024,
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"You are a professional translator. Translate the following English text into {lang_name}. "
                        f"Preserve all technical terms, jargon, and fabricated terminology exactly as they sound "
                        f"(transliterate if needed). Maintain the same tone, formality, and question structure. "
                        f"Output ONLY the translation, nothing else."
                    ),
                },
                {"role": "user", "content": question_text},
            ],
        )
        return resp.choices[0].message.content.strip()


async def main():
    # Load BullshitBench
    with open("datasets/bullshitbench_v2_questions.json") as f:
        questions = json.load(f)

    print(f"Loaded {len(questions)} BullshitBench questions")

    # English originals
    translated = {"en": questions}

    for lang_code, lang_name in LANGUAGES.items():
        print(f"\nTranslating to {lang_name} ({lang_code})...")
        tasks = [
            translate_question(q["question"], lang_code, lang_name)
            for q in questions
        ]
        translations = await tqdm_asyncio.gather(*tasks, desc=lang_name)

        lang_questions = []
        for orig, trans in zip(questions, translations):
            lang_questions.append({
                "id": orig["id"],
                "question_en": orig["question"],
                "question": trans,
                "nonsensical_element": orig["nonsensical_element"],
                "domain": orig["domain"],
                "technique": orig.get("technique", ""),
                "language": lang_code,
            })
        translated[lang_code] = lang_questions
        print(f"  Completed {len(lang_questions)} translations")

    # Save
    output_path = "datasets/bullshitbench_multilingual.json"
    with open(output_path, "w") as f:
        json.dump(translated, f, ensure_ascii=False, indent=2)
    print(f"\nSaved multilingual BullshitBench to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
