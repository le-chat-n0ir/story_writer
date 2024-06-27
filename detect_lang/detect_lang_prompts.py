from typing import Final

from langchain_core.prompts import ChatPromptTemplate

from structure.prompts import USER, ChatRole, HistoryKey

LANGUAGE: Final[str] = "language"

detect_lang_prompt = ChatPromptTemplate.from_messages(
    [
        (
            ChatRole.SYSTEM,
            "You are an AI assistant for language detection tasks.\n"
            "Your only task is to detect if a text is written in english.\n"
            f"Return the JSON with a single key '{LANGUAGE}' with no preamble or explanation.\n"
            f"If the text is in english, return '{LANGUAGE}': 'en'.\n"
            f"If the text is not in english, return '{LANGUAGE}': 'not_en'.\n"
            f"",
        ),
        # (
        #     USER,
        #     f"Die Sonne scheint hell und warm. "
        #     f"Ein Schmetterling fliegt über eine Wiese, und setzt sich zärtlich auf eine Blume.",
        # ),
        # (
        #     AI,
        #     f"The sun is shining bright and warm. "
        #     f"A butterfly flies over a meadow, and gently settles on a flower.",
        # ),
        # (
        #     PLACEHOLDER,
        #     f"{{{HISTORY}}}",
        # ),
        (
            USER,
            f"{{{HistoryKey.INPUT}}}",
        ),
    ]
)
