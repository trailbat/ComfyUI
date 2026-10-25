import json

ANY = "*"


def lookup(data, key):
    """Follows a dotted key such as "prompt", "lora.strength" or "loras.0.name". Returns None when it isn't there."""
    for part in key.split("."):
        if isinstance(data, dict):
            data = data.get(part)
        elif isinstance(data, (list, tuple)) and part.lstrip("-").isdigit() and -len(data) <= int(part) < len(data):
            data = data[int(part)]
        else:
            return None
    return data


def to_number(value, kind):
    try:
        return kind(float(value)) if kind is int else kind(value)
    except (TypeError, ValueError):
        return kind(0)


class GetValue:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "data": (ANY, {"tooltip": "A dict such as ▶ For Each's item when looping over a prompt list."}),
                "key": ("STRING", {"default": "prompt", "tooltip": "The field to read, use dots for nested fields, e.g. \"lora.strength\" or \"loras.0.name\"."}),
                "default": ("STRING", {"default": "", "tooltip": "Used when the field is missing, e.g. 7 for a cfg the YAML file doesn't set."}),
            }
        }

    RETURN_TYPES = ("STRING", "INT", "FLOAT", "BOOLEAN")
    RETURN_NAMES = ("string", "int", "float", "boolean")
    OUTPUT_TOOLTIPS = ("Lists and mappings come out as JSON.", "0 when the value isn't a number.",
                       "0.0 when the value isn't a number.", "True for true, yes, on and non-zero numbers.")
    FUNCTION = "get"
    CATEGORY = "utils/personal"
    DESCRIPTION = "Reads one field as a string, int, float and boolean, connect whichever output you need."

    def get(self, data, key, default):
        value = lookup(data, key.strip())
        if value is None:
            value = default
        text = json.dumps(value) if isinstance(value, (dict, list)) else str(value)
        boolean = value if isinstance(value, bool) else text.strip().lower() in ("true", "yes", "on") or to_number(value, float) != 0
        return (text, to_number(value, int), to_number(value, float), boolean)
