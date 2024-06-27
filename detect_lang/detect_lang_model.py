from langchain_core.output_parsers import JsonOutputParser

from detect_lang import detect_lang_prompts
from structure import models

chain = detect_lang_prompts.detect_lang_prompt | models.json_model | JsonOutputParser()
