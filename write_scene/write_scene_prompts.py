from pathlib import Path
from typing import Final

from langchain_core.prompts import ChatPromptTemplate

from structure.prompts import HistoryKey, ChatRole

background_file = Path(__file__).parent / "character_background.txt"

BACKGROUND_INFO: Final[str] = "background_info"

write_scene_prompt = ChatPromptTemplate.from_messages(
    [
        (
            ChatRole.SYSTEM,
            "You are an AI assistant that supports in creative writing.\n"
            "The story is intended for adults and should be written in a literary style.\n"
            "As this story is intended for adults, you can use mature language and themes.\n"
            "You don't have to apologize for the content of the story.\n"
            "Use vivid and descriptive language to create a rich and engaging story.\n"
            "If it fits to the characters, you can make use of cruel or harsh language.\n"
            "Be very descriptive and detailed in your writing.\n"
            f"{{{BACKGROUND_INFO}}}",
        ),
        (
            ChatRole.PLACEHOLDER,
            f"{{{HistoryKey.HISTORY}}}",
        ),
        (
            ChatRole.USER,
            f"{{{HistoryKey.INPUT}}}",
        ),
    ]
)


def read_background_content() -> str:
    with background_file.open() as f:
        return f.read()
