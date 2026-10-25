# Prompt Picker: a list of named prompts with a checkbox each. Every ticked prompt comes out as its own item of a list
# output, so the nodes after it run once per prompt and one Run makes an image for each.
#
# The prompts live in the node's text box, each one under a header line, "[x]" when ticked and "[ ]" when not:
#
#   [x] Lighthouse
#   A lighthouse on a cliff at dusk, waves crashing below.
#
#   [ ] Cat
#   A ginger cat asleep on a windowsill.
#
# js/prompt_picker.js shows the headers as a checklist above the text box, ticking one there edits its header.

import re

WEB_DIRECTORY = "./js"

HEADER = re.compile(r"^\s*\[([ xX]?)\]\s*(.*?)\s*$")

DEFAULT_PROMPTS = """[x] Lighthouse
A lighthouse on a rocky cliff at dusk, waves crashing below, warm light in the lamp room, dramatic clouds.

[x] Ramen shop
A tiny ramen shop at night in a rainy Tokyo alley, steam rising from the counter, neon signs reflected in puddles.

[ ] Fox in snow
A red fox standing in deep fresh snow in a birch forest, soft morning light, shallow depth of field.
"""


def parse_prompts(text):
    """Returns [(name, ticked, prompt)] in the order they appear, text before the first header is ignored."""
    prompts = []
    for line in text.splitlines():
        match = HEADER.match(line)
        if match:
            prompts.append([match.group(2) or f"Prompt {len(prompts) + 1}", match.group(1).lower() == "x", []])
        elif prompts:
            prompts[-1][2].append(line)
    return [(name, ticked, "\n".join(body).strip()) for name, ticked, body in prompts]


def file_name(name):
    return re.sub(r"[^\w\- ]+", "_", name).strip() or "prompt"


class PromptPicker:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompts": ("STRING", {"multiline": True, "default": DEFAULT_PROMPTS,
                                       "tooltip": "One prompt per header line: \"[x] Name\" when ticked, \"[ ] Name\" when not, the prompt text on the lines below it."}),
                "prefix": ("STRING", {"default": "",
                                      "tooltip": "Added before every prompt, e.g. a LoRA trigger word. Leave empty for none."}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompt", "name")
    OUTPUT_IS_LIST = (True, True)
    OUTPUT_TOOLTIPS = ("The ticked prompts, the nodes after this run once for each.", "The ticked prompts' names, usable as a file name.")
    FUNCTION = "pick"
    CATEGORY = "utils/prompt"
    DESCRIPTION = "Runs the nodes after it once for each ticked prompt."

    def pick(self, prompts, prefix):
        picked = [(name, text) for name, ticked, text in parse_prompts(prompts) if ticked and text]
        if not picked:
            raise ValueError("Prompt Picker: no prompts are ticked")
        prefix = prefix.strip()
        texts = [f"{prefix} {text}" if prefix else text for _, text in picked]
        return (texts, [file_name(name) for name, _ in picked])


NODE_CLASS_MAPPINGS = {"PromptPicker": PromptPicker}
NODE_DISPLAY_NAME_MAPPINGS = {"PromptPicker": "Prompt Picker"}
