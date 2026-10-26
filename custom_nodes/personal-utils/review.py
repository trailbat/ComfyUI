# Pending Prompts drops the prompts that already have an image in the output folder or are unticked in the Reference
# Picker, so a run only makes what's missing. The routes let js/reference_picker.js list the generated images and move
# rejected ones to <folder>/rejected, which makes their prompt pending again.

import json
import os
import re

from aiohttp import web

import folder_paths
from server import PromptServer

# Image Save names the files <filename_prefix>_0001.png, the prefix being the prompt id.
RESULT_NAME = re.compile(r"^(.+)_\d+\.[^.]+$")


def output_path(folder):
    """Absolute path of a subfolder of the output directory, refusing anything that escapes it."""
    base = os.path.abspath(folder_paths.get_output_directory())
    path = os.path.abspath(os.path.join(base, folder.strip().strip("/\\")))
    if os.path.commonpath((base, path)) != base:
        raise ValueError(f"Not inside the output folder: {folder!r}")
    return path


def list_results(folder):
    """The generated images in output/<folder> as (file name, prompt id)."""
    directory = output_path(folder)
    if not os.path.isdir(directory):
        return []
    files = sorted(f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)))
    return [(f, m.group(1)) for f in folder_paths.filter_files_content_types(files, ["image"]) if (m := RESULT_NAME.match(f))]


class PendingPrompts:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompts": ("JSON_LIST", {"tooltip": "Load Prompt List's prompts, each with an id."}),
                "folder": ("STRING", {"default": "lora_dataset", "tooltip": "The subfolder of output the images are saved to, Image Save's output_path."}),
                "skip": ("STRING", {"multiline": True, "default": "",
                                    "tooltip": "Prompt ids not to run, one per line. Untick them in the Reference Picker's prompt list."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("JSON_LIST", "INT")
    RETURN_NAMES = ("prompts", "count")
    OUTPUT_TOOLTIPS = ("The prompts left to generate, connect it to ▶ For Each.", "The number of prompts left.")
    FUNCTION = "pending"
    CATEGORY = "utils/personal"
    DESCRIPTION = "Keeps the ticked prompts that have no image in the output folder yet, rejected images don't count."

    def pending(self, prompts, folder, skip, unique_id):
        done = {pid for _, pid in list_results(folder)}
        skipped = {s.strip() for s in skip.splitlines()}
        pending = [p for i, p in enumerate(prompts) if str(p.get("id", i + 1)) not in done | skipped]
        PromptServer.instance.send_progress_text(f"{len(pending)} of {len(prompts)} to generate", unique_id)
        return (pending, len(pending))

    @classmethod
    def IS_CHANGED(cls, folder, **kwargs):
        return json.dumps(list_results(folder))


routes = PromptServer.instance.routes


@routes.get("/personal-utils/results")
async def get_results(request):
    folder = request.query.get("folder", "")
    directory = output_path(folder)
    return web.json_response([{"file": f, "id": pid, "mtime": os.path.getmtime(os.path.join(directory, f))} for f, pid in list_results(folder)])


@routes.post("/personal-utils/reject")
async def reject_result(request):
    body = await request.json()
    folder, file = body["folder"], body["file"]
    if file not in {f for f, _ in list_results(folder)}:
        return web.Response(status=404, text=f"No result {file!r} in output/{folder}")
    directory = output_path(folder)
    os.makedirs(os.path.join(directory, "rejected"), exist_ok=True)
    os.replace(os.path.join(directory, file), os.path.join(directory, "rejected", file))
    return web.json_response({})
