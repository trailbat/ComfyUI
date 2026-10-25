# Personal utility nodes:
#   Load Prompt List (YAML)  - reads a YAML file from the input folder into a list of prompt dicts, see example_prompts.yaml
#   ▶ For Each / For Each ◀  - runs the nodes between them once per item of a list and collects what reaches the end node
#   Get Value                - reads one field of a dict (e.g. a prompt from the list) as STRING, INT, FLOAT and BOOLEAN
#
# js/upload.js adds an upload button to the YAML loader.

from .loop import ForEachBegin, ForEachEnd
from .prompts import LoadPromptList
from .values import GetValue

WEB_DIRECTORY = "./js"

NODE_CLASS_MAPPINGS = {
    "PersonalLoadPromptList": LoadPromptList,
    "PersonalForEachBegin": ForEachBegin,
    "PersonalForEachEnd": ForEachEnd,
    "PersonalGetValue": GetValue,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PersonalLoadPromptList": "Load Prompt List (YAML)",
    "PersonalForEachBegin": "▶ For Each",
    "PersonalForEachEnd": "For Each ◀",
    "PersonalGetValue": "Get Value",
}
