import hashlib
import os

import yaml

import folder_paths

EXTENSIONS = (".yaml", ".yml")


def parse_prompt_list(text):
    """Returns one dict per prompt with the file's defaults filled in. The file is either a list of prompts or a mapping
    with "prompts" and an optional "defaults" mapping. A prompt is a mapping or a plain string, short for {prompt: ...}."""
    data = yaml.safe_load(text)
    if isinstance(data, list):
        data = {"prompts": data}
    if not isinstance(data, dict) or not isinstance(data.get("prompts"), list):
        raise ValueError("Prompt list YAML needs a top level list of prompts or a \"prompts:\" list")
    defaults = data.get("defaults") or {}
    if not isinstance(defaults, dict):
        raise ValueError("Prompt list YAML: \"defaults:\" must be a mapping of setting names to values")

    prompts = []
    for i, item in enumerate(data["prompts"]):
        if isinstance(item, str):
            item = {"prompt": item}
        if not isinstance(item, dict):
            raise ValueError(f"Prompt list YAML: prompt {i + 1} must be a string or a mapping, got {item!r}")
        prompts.append({**defaults, **item})
    return prompts


class LoadPromptList:
    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {
            "required": {
                "file": (sorted(folder_paths.filter_files_extensions(files, EXTENSIONS)),
                         {"tooltip": "A .yaml file in the input folder, use the upload button to add one."}),
            }
        }

    RETURN_TYPES = ("JSON_LIST", "INT")
    RETURN_NAMES = ("prompts", "count")
    OUTPUT_TOOLTIPS = ("One dict per prompt, connect it to ▶ For Each.", "The number of prompts.")
    FUNCTION = "load"
    CATEGORY = "utils/personal"
    DESCRIPTION = "Loads a list of prompts and their settings (cfg, steps, ...) from a YAML file."

    def load(self, file):
        with open(self.path(file), encoding="utf-8") as f:
            prompts = parse_prompt_list(f.read())
        return (prompts, len(prompts))

    @staticmethod
    def path(file):
        if not file.lower().endswith(EXTENSIONS):
            raise ValueError(f"Not a YAML file: {file!r}")
        return folder_paths.get_annotated_filepath(file)

    @classmethod
    def IS_CHANGED(cls, file):
        with open(cls.path(file), "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()

    @classmethod
    def VALIDATE_INPUTS(cls, file):
        if not folder_paths.exists_annotated_filepath(file):
            return f"Prompt list file not found: {file}"
        return True
