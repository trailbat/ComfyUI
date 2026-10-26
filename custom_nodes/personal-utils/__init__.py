# Personal utility nodes:
#   Load Prompt List (YAML)  - reads a YAML file from the input folder into a list of prompt dicts, see example_prompts.yaml
#   ▶ For Each / For Each ◀  - runs the nodes between them once per item of a list and collects what reaches the end node
#   Get Value                - reads one field of a dict (e.g. a prompt from the list) as STRING, INT, FLOAT and BOOLEAN
#   Reference Analyzer       - scores the photos in an input subfolder against every prompt of a prompt list
#   Reference Picker         - the photos picked for the loop's current prompt, picked in its grid from the analysis
#   Pending Prompts          - the prompts without an image in the output folder yet, reviewed in the picker's Results
#   LoRA Sliders             - applies the slider LoRAs of the Colab notebooks' models.yaml that are ticked in its list
#   LoRA Styles              - mixes the style LoRAs of models.yaml that are ticked in its list, with their preview photos
#
# js/upload.js adds an upload button to the YAML loader, js/reference_picker.js the Reference Picker's grid and
# js/lora_sliders.js the LoRA list with its browse dialog, for LoRA Sliders and LoRA Styles.
# lora_dataset_prompts.yaml is the LoRA dataset prompt list with the Reference Analyzer's targets.

from .lora_sliders import LoraSliders
from .lora_styles import LoraStyles
from .loop import ForEachBegin, ForEachEnd
from .prompts import LoadPromptList
from .references import ReferenceAnalyzer, ReferencePicker
from .review import PendingPrompts
from .values import GetValue

WEB_DIRECTORY = "./js"

NODE_CLASS_MAPPINGS = {
    "PersonalLoadPromptList": LoadPromptList,
    "PersonalForEachBegin": ForEachBegin,
    "PersonalForEachEnd": ForEachEnd,
    "PersonalGetValue": GetValue,
    "PersonalReferenceAnalyzer": ReferenceAnalyzer,
    "PersonalReferencePicker": ReferencePicker,
    "PersonalPendingPrompts": PendingPrompts,
    "PersonalLoraSliders": LoraSliders,
    "PersonalLoraStyles": LoraStyles,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PersonalLoadPromptList": "Load Prompt List (YAML)",
    "PersonalForEachBegin": "▶ For Each",
    "PersonalForEachEnd": "For Each ◀",
    "PersonalGetValue": "Get Value",
    "PersonalReferenceAnalyzer": "Reference Analyzer",
    "PersonalReferencePicker": "Reference Picker",
    "PersonalPendingPrompts": "Pending Prompts",
    "PersonalLoraSliders": "LoRA Sliders",
    "PersonalLoraStyles": "LoRA Styles",
}
