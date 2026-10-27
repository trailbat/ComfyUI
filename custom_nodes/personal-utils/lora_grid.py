# LoRA Grid Plan lists every ticked LoRA at every ticked weight, for ▶ For Each to run one image each. LoRA Grid puts the
# images of the loop back together: the weight is drawn on each image, the images of a LoRA are tiled into one picture
# and the LoRA's name is drawn on it. js/lora_grid.js has the two lists of the plan node.

import itertools
import json
import os

import torch

from comfy_extras.nodes_text_overlay import TextOverlay

PLAN = "LORA_GRID_PLAN"
WEIGHT_SIZE = 6.0  # percent of an image's height
NAME_SIZE = 8.0  # percent of an image's height


class LoraGridPlan:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "loras": ("STRING", {"default": "[]", "tooltip": "The list of LoRAs, edited in the node."}),
                "weights": ("STRING", {"default": "[]", "tooltip": "The list of weights, edited in the node."}),
            },
        }

    RETURN_TYPES = (PLAN,)
    RETURN_NAMES = ("plan",)
    OUTPUT_IS_LIST = (True,)
    OUTPUT_TOOLTIPS = ("One entry per ticked LoRA and ticked weight, connect to ▶ For Each and to LoRA Grid.",)
    FUNCTION = "plan"
    CATEGORY = "utils/personal"
    DESCRIPTION = "Every ticked LoRA at every ticked weight, one LoRA after the other."

    def plan(self, loras, weights):
        names = [row["name"] for row in json.loads(loras) if row["enabled"]]
        values = [row["weight"] for row in json.loads(weights) if row["enabled"]]
        if not names or not values:
            raise ValueError("LoRA Grid Plan: tick at least one LoRA and one weight")
        return ([{"lora": name, "weight": weight} for name in names for weight in values],)


class LoraGrid:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The images of the loop, For Each ◀'s batch."}),
                "plan": (PLAN, {"tooltip": "The plan that ▶ For Each ran."}),
                "columns": ("INT", {"default": 0, "min": 0, "max": 64, "tooltip": "Images per row of a LoRA's picture, 0 puts all its weights in one row."}),
            },
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("IMAGE",)
    OUTPUT_TOOLTIPS = ("One picture per LoRA.",)
    FUNCTION = "tile"
    CATEGORY = "utils/personal"
    DESCRIPTION = "Draws the weight on every image, tiles the images of each LoRA into one picture and draws the LoRA's name on it."

    def tile(self, images, plan, columns):
        images, columns = images[0], columns[0]
        labelled = [TextOverlay.execute(image.unsqueeze(0), f"{item['weight']:g}", WEIGHT_SIZE, "#ffffff", "top", "left", True)[0]
                    for image, item in zip(images, plan)]
        height, width, channels = images.shape[1:]
        grids = []
        start = 0
        for lora, group in itertools.groupby(plan, key=lambda item: item["lora"]):
            count = len(list(group))
            across = min(columns, count) if columns else count
            down = -(-count // across)
            tiles = torch.cat(labelled[start:start + count])
            tiles = torch.cat([tiles, tiles.new_zeros(down * across - count, height, width, channels)])
            grid = tiles.reshape(down, across, height, width, channels).permute(0, 2, 1, 3, 4).reshape(1, down * height, across * width, channels)
            grids.append(TextOverlay.execute(grid, os.path.splitext(lora)[0], NAME_SIZE / down, "#ffffff", "bottom", "center", True)[0])
            start += count
        return (torch.cat(grids),)
