from langchain_core.output_parsers import JsonOutputParser

from structure import models
from write_scene.write_scene_prompts import write_scene_prompt

translate_chain = write_scene_prompt | models.json_model | JsonOutputParser()
