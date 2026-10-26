# LoRA Styles mixes the style LoRAs ticked in its list: style: true entries of the loras folder in the Colab notebooks'
# models.yaml, each at its strength times the node's strength, with the preview photos (previews: image urls) the catalog
# gives them. The list and the browse dialog are js/lora_sliders.js, shared with LoRA Sliders.

import os

from aiohttp import web

from server import PromptServer

from .lora_sliders import ALL, apply_listed, catalog_loras, catalog_path, lora_infos


class LoraStyles:
    @classmethod
    def INPUT_TYPES(cls):
        model_types = sorted({lora["model"] for lora in catalog_loras("style").values() if "model" in lora})
        return {
            "required": {
                "model": ("MODEL",),
                "model_type": ([ALL] + model_types, {"tooltip": "The model family the styles are for, the list and the browse dialog show only these."}),
                "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05, "tooltip": "Scales the strength of every ticked style."}),
                "loras": ("STRING", {"default": "[]", "tooltip": "The list of styles with their strengths, edited in the node."}),
            },
            "optional": {
                "clip": ("CLIP",),
                "prompt": ("STRING", {"forceInput": True, "tooltip": "The prompt the ticked trigger words are added to the end of."}),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "STRING")
    RETURN_NAMES = ("model", "clip", "prompt")
    OUTPUT_TOOLTIPS = ("The model with the ticked styles applied.", "The clip with the ticked styles applied, if connected.",
                       "The prompt with the trigger words of the ticked styles added to the end.")
    FUNCTION = "apply"
    CATEGORY = "utils/personal"
    DESCRIPTION = "Mixes the style LoRAs of the catalog that are ticked in the list, each at its strength times the node's strength, and adds their trigger words to the prompt."

    def apply(self, model, model_type, strength, loras, clip=None, prompt=""):
        return apply_listed("LoRA Styles", "style", model, clip, model_type, loras, prompt, strength)

    @classmethod
    def IS_CHANGED(cls, model_type, strength, loras, **kwargs):
        return f"{model_type}{strength}{loras}{os.path.getmtime(catalog_path())}"


@PromptServer.instance.routes.get("/personal-utils/styles")
async def get_styles(request):
    return web.json_response(lora_infos("style"))
