# LoRA Sliders applies the LoRAs ticked in its list at their strengths. The list is the slider: true entries of the
# loras folder in the Colab notebooks' models.yaml, which carry a description and the range they work in. The catalog
# is the sibling ComfyUI_Colab_Notebooks checkout when there is one, else colab/models.yaml, which the Colab setup
# downloads from GitHub. COMFY_LORA_CATALOG overrides both. js/lora_sliders.js has the list and the browse dialog.

import json
import os
import sys

from aiohttp import web

import comfy.sd
import comfy.utils
import folder_paths
from server import PromptServer

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
COLAB_DIR = os.path.join(ROOT, "colab")
sys.path.append(COLAB_DIR)
from catalog import parse_catalog  # noqa: E402

ALL = "All"
CATALOGS = (os.path.join(os.path.dirname(ROOT), "ComfyUI_Colab_Notebooks", "models.yaml"), os.path.join(COLAB_DIR, "models.yaml"))


def catalog_path():
    override = os.environ.get("COMFY_LORA_CATALOG")
    return override or next((p for p in CATALOGS if os.path.isfile(p)), CATALOGS[-1])


def catalog_loras(flag):
    """The LoRAs of the catalog with flag (slider or style) set, by name."""
    with open(catalog_path(), encoding="utf-8") as f:
        catalog = parse_catalog(f.read())
    return {name: lora for name, lora in catalog.items() if lora["folder"] == "loras" and lora.get(flag)}


def trigger_words(lora):
    """The trigger_words of a catalog entry, a string or a list, as one string."""
    words = lora.get("trigger_words", "")
    return ", ".join(words) if isinstance(words, list) else words.strip()


def in_model_type(lora, model_type):
    return model_type == ALL or lora.get("model") == model_type


def apply_listed(title, flag, model, clip, model_type, loras, prompt, scale=1.0):
    """Applies the ticked rows of a node's list at their strength times scale, returns model, clip and the prompt with the trigger words."""
    available = catalog_loras(flag)
    words = []
    for row in json.loads(loras):
        if not row["enabled"]:
            continue
        lora = available.get(row["name"])
        if lora is None:
            raise ValueError(f"{title}: {row['name']!r} isn't a {flag} LoRA of the catalog {catalog_path()}")
        if not in_model_type(lora, model_type):
            continue
        if row.get("triggers"):
            words.append(trigger_words(lora))
        path = folder_paths.get_full_path("loras", lora["filename"])
        if path is None:
            raise FileNotFoundError(f"{title}: {lora['filename']} isn't in the loras folder, install {row['name']!r} in the Download Models tab")
        strength = row["strength"] * scale
        model, clip = comfy.sd.load_lora_for_models(model, clip, comfy.utils.load_torch_file(path, safe_load=True), strength, strength)
    return model, clip, ", ".join(filter(None, [prompt.strip().rstrip(",")] + words))


def lora_infos(flag):
    return [{
        "name": name,
        "model": lora.get("model", ""),
        "description": lora.get("description", ""),
        "trigger_words": trigger_words(lora),
        "author": lora.get("author", ""),
        "page": lora.get("page", ""),
        "previews": lora.get("previews", []),
        "min": lora.get("min", 0.0),
        "max": lora.get("max", 1.0),
        "default": lora.get("default", 1.0),
        "installed": folder_paths.get_full_path("loras", lora["filename"]) is not None,
    } for name, lora in catalog_loras(flag).items()]


class LoraSliders:
    @classmethod
    def INPUT_TYPES(cls):
        model_types = sorted({lora["model"] for lora in catalog_loras("slider").values() if "model" in lora})
        return {
            "required": {
                "model": ("MODEL",),
                "model_type": ([ALL] + model_types, {"tooltip": "The model family the LoRAs are for, the list and the browse dialog show only these."}),
                "loras": ("STRING", {"default": "[]", "tooltip": "The list of LoRAs with their strengths, edited in the node."}),
            },
            "optional": {
                "clip": ("CLIP",),
                "prompt": ("STRING", {"forceInput": True, "tooltip": "The prompt the ticked trigger words are added to the end of."}),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "STRING")
    RETURN_NAMES = ("model", "clip", "prompt")
    OUTPUT_TOOLTIPS = ("The model with the ticked LoRAs applied.", "The clip with the ticked LoRAs applied, if connected.",
                       "The prompt with the trigger words of the ticked LoRAs added to the end.")
    FUNCTION = "apply"
    CATEGORY = "utils/personal"
    DESCRIPTION = "Applies the slider LoRAs of the catalog that are ticked in the list, each at its strength, and adds their trigger words to the prompt."

    def apply(self, model, model_type, loras, clip=None, prompt=""):
        return apply_listed("LoRA Sliders", "slider", model, clip, model_type, loras, prompt)

    @classmethod
    def IS_CHANGED(cls, model_type, loras, **kwargs):
        return f"{model_type}{loras}{os.path.getmtime(catalog_path())}"


routes = PromptServer.instance.routes


@routes.get("/personal-utils/loras")
async def get_loras(request):
    return web.json_response(lora_infos("slider"))
