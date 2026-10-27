# Personal utility nodes:
#   Load Prompt List (YAML)  - reads a YAML file from the input folder into a list of prompt dicts, see example_prompts.yaml
#   ▶ For Each / For Each ◀  - runs the nodes between them once per item of a list and collects what reaches the end node
#   Get Value                - reads one field of a dict (e.g. a prompt from the list) as STRING, INT, FLOAT and BOOLEAN
#   Reference Analyzer       - scores the photos in an input subfolder against every prompt of a prompt list
#   Reference Picker         - the photos picked for the loop's current prompt, picked in its grid from the analysis
#   Pending Prompts          - the prompts without an image in the output folder yet, reviewed in the picker's Results
#   LoRA Sliders             - applies the slider LoRAs of the Colab notebooks' models.yaml that are ticked in its list
#   LoRA Styles              - mixes the style LoRAs of models.yaml that are ticked in its list, with their preview photos
#   LoRA Grid Plan           - every ticked LoRA at every ticked weight, in lists with ticks, for ▶ For Each
#   Find People              - groups the faces in a folder of photos by person and previews them
#   Select Person            - picks the person in every photo, with corrections for the face recognition's mistakes
#   Review People            - every photo with the picked face boxed and the faces numbered
#   Save People              - writes who is in every photo to output/people, for another workflow
#   Load People              - reads that file back
#   Plan Captions            - plans every photo's caption opening (trigger, sex, where the person is among the others)
#   Load Item Image          - the current photo, and a copy with the person's face boxed
#   Save Caption             - writes the caption next to the photo as a .txt file
#   LoRA Grid                - tiles the images of that loop into one picture per LoRA, with the weights and the name drawn on
#
# js/upload.js adds an upload button to the YAML loader, js/reference_picker.js the Reference Picker's grid and
# js/lora_sliders.js the LoRA list with its browse dialog, for LoRA Sliders and LoRA Styles, and js/lora_grid.js the lists
# of LoRA Grid Plan.
# lora_dataset_prompts.yaml is the LoRA dataset prompt list with the Reference Analyzer's targets.

from .captions import FindPeople, LoadItemImage, LoadPeople, PlanCaptions, ReviewPeople, SaveCaption, SavePeople, SelectPerson
from .lora_grid import LoraGrid, LoraGridPlan
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
    "PersonalLoraGridPlan": LoraGridPlan,
    "PersonalLoraGrid": LoraGrid,
    "PersonalFindPeople": FindPeople,
    "PersonalSelectPerson": SelectPerson,
    "PersonalReviewPeople": ReviewPeople,
    "PersonalSavePeople": SavePeople,
    "PersonalLoadPeople": LoadPeople,
    "PersonalPlanCaptions": PlanCaptions,
    "PersonalLoadItemImage": LoadItemImage,
    "PersonalSaveCaption": SaveCaption,
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
    "PersonalLoraGridPlan": "LoRA Grid Plan",
    "PersonalLoraGrid": "LoRA Grid",
    "PersonalFindPeople": "Find People",
    "PersonalSelectPerson": "Select Person",
    "PersonalReviewPeople": "Review People",
    "PersonalSavePeople": "Save People",
    "PersonalLoadPeople": "Load People",
    "PersonalPlanCaptions": "Plan Captions",
    "PersonalLoadItemImage": "Load Item Image",
    "PersonalSaveCaption": "Save Caption",
}
