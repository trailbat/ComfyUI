# Find People groups the faces of a folder of photos by person and shows one face per person, Select Person picks one of
# them and takes corrections for the photos where the face recognition was wrong, Review People shows who was picked in
# every photo and Save People writes the result to output/people. Load People reads it back for Plan Captions, which
# plans the caption of every photo: the opening with the trigger and where that person is among the others. Load Item
# Image and Save Caption are the two ends of the loop that captions one photo. See example_workflows/lora-caption-*.json.
# Faces are numbered from the left of the photo, photos from 1 in folder order.

import hashlib
import json
import math
import os

import numpy as np
import torch
from insightface.app import FaceAnalysis
from PIL import Image, ImageDraw, ImageFont, ImageOps

import comfy.model_management
import comfy.utils
import folder_paths
import node_helpers

from .references import INSIGHTFACE_DIR, normalize
from .review import output_path

ANY = "*"

DETECT_SIZE = 2048  # longest side the face detector sees
SAME_PERSON = 0.4  # cosine similarity of two w600k_r50 embeddings above which faces are one person
DETECT_SIZES = (640, 1024)  # the detector misses faces that are too big or too small for its input size, so photos go through both
SAME_FACE = 0.4  # overlap above which two detections are one face
MIN_FACE = 0.03  # smallest face height, as a fraction of the photo's, that counts as a person
FRONT_RATIO = 1.6  # face height ratio above which a person is in the front or the back
TILE = 192
REVIEW_TILE = 256
REVIEW_COLUMNS = 6
PEOPLE_FOLDER = "people"  # in the output folder
SHEET_PEOPLE = 12
SHEET_COLUMNS = 4


def folder_path(folder):
    return os.path.abspath(folder.strip().strip('"'))


def list_images(folder):
    if not os.path.isdir(folder):
        raise ValueError(f"Not a folder: {folder}")
    files = sorted(f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f)))
    return folder_paths.filter_files_content_types(files, ["image"])


def caption_path(image_path):
    return os.path.splitext(image_path)[0] + ".txt"


def load_rgb(path):
    image = node_helpers.pillow(Image.open, path)
    return node_helpers.pillow(ImageOps.exif_transpose, image).convert("RGB")


def to_tensor(image):
    pixels = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0)[None]
    return pixels.to(device=comfy.model_management.intermediate_device(), dtype=comfy.model_management.intermediate_dtype())


def face_height(face):
    return face["box"][3] - face["box"][1]


def overlap(a, b):
    width, height = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    inner = max(0.0, width) * max(0.0, height)
    return inner / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inner)


def merge_faces(faces):
    """The faces of all detector passes without doubles, the surest detection of each, from the left of the photo."""
    kept = []
    for face in sorted(faces, key=lambda f: -f["score"]):
        if all(overlap(face["box"], k["box"]) < SAME_FACE for k in kept):
            kept.append(face)
    return sorted(kept, key=lambda f: f["box"][0])


def crop_face(photo, box):
    """The tile for a face, and the function that maps a normalized box of the photo onto the tile."""
    x0, y0, x1, y1 = box[0] * photo.width, box[1] * photo.height, box[2] * photo.width, box[3] * photo.height
    side = 2.4 * max(x1 - x0, y1 - y0)
    left, top = (x0 + x1) / 2 - side / 2, (y0 + y1) / 2 - side / 2
    tile = photo.crop((round(left), round(top), round(left + side), round(top + side))).resize((TILE, TILE), Image.LANCZOS)
    scale = TILE / side
    return tile, lambda b: ((b[0] * photo.width - left) * scale, (b[1] * photo.height - top) * scale,
                            (b[2] * photo.width - left) * scale, (b[3] * photo.height - top) * scale)


def position(face, others):
    """Where the person is among the others, from the face boxes: in the front or back when their face is much bigger or
    smaller than every other one, otherwise on the left or right of the frame."""
    if face is None or not others:
        return ""
    biggest = max(face_height(o) for o in others)
    if face_height(face) >= FRONT_RATIO * biggest:
        return "in the front"
    if biggest >= FRONT_RATIO * face_height(face):
        return "in the back"
    x = lambda f: f["box"][0] + f["box"][2]
    if all(x(face) < x(o) for o in others):
        return "on the left"
    if all(x(face) > x(o) for o in others):
        return "on the right"
    return "in the middle"


class FindPeople:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder": ("STRING", {"default": "", "tooltip": "The folder with the photos, e.g. C:\\datasets\\my_subject."}),
                "min_confidence": ("FLOAT", {"default": 0.6, "min": 0.2, "max": 0.95, "step": 0.05,
                                             "tooltip": "How sure the detector has to be that something is a face. Lower it when faces in Review People have no box, "
                                                        "pets and objects may get a box then."}),
            }
        }

    RETURN_TYPES = ("PEOPLE", "IMAGE")
    RETURN_NAMES = ("people", "preview")
    OUTPUT_TOOLTIPS = ("Connect to Select Person.", "One face per person, the most frequent first, connect to Preview Image to pick the person.")
    FUNCTION = "find"
    CATEGORY = "utils/personal"
    DESCRIPTION = "Finds the faces in every photo of the folder and groups them by person."

    def find(self, folder, min_confidence):
        folder = folder_path(folder)
        names = list_images(folder)
        if not names:
            raise ValueError(f"Find People: no images in {folder}")
        detectors = [FaceAnalysis(name=INSIGHTFACE_DIR, allowed_modules=["detection", "recognition"]) for _ in DETECT_SIZES]
        for detector, size in zip(detectors, DETECT_SIZES):
            detector.prepare(ctx_id=0, det_size=(size, size), det_thresh=min_confidence)
        progress = comfy.utils.ProgressBar(len(names))
        photos = []
        for name in names:
            photo = load_rgb(os.path.join(folder, name))
            scale = min(1.0, DETECT_SIZE / max(photo.size))
            if scale < 1:
                photo = photo.resize((round(photo.width * scale), round(photo.height * scale)), Image.LANCZOS)
            bgr = np.ascontiguousarray(np.asarray(photo)[:, :, ::-1])
            box_scale = np.array([photo.width, photo.height] * 2)
            faces = [{"box": [float(v) for v in f.bbox / box_scale], "score": float(f.det_score), "embedding": normalize(f.embedding.astype(np.float32))}
                     for detector in detectors for f in detector.get(bgr)]
            photos.append({"name": name, "path": os.path.join(folder, name), "faces": merge_faces([f for f in faces if face_height(f) > MIN_FACE])})
            progress.update(1)
        del detectors

        # Biggest and surest faces first so they set the centers, a person is never twice in one photo.
        faces = sorted(((i, f) for i, p in enumerate(photos) for f in p["faces"]), key=lambda t: -t[1]["score"] * face_height(t[1]))
        clusters = []
        for i, face in faces:
            best, best_similarity = None, SAME_PERSON
            for cluster in clusters:
                similarity = float(face["embedding"] @ cluster["center"])
                if i not in cluster["photos"] and similarity >= best_similarity:
                    best, best_similarity = cluster, similarity
            if best is None:
                best = {"total": np.zeros_like(face["embedding"]), "photos": set(), "faces": []}
                clusters.append(best)
            best["total"] += face["embedding"]
            best["center"] = normalize(best["total"])
            best["photos"].add(i)
            best["faces"].append((i, face))
        if not clusters:
            raise ValueError(f"Find People: no faces found in {folder}")
        clusters.sort(key=lambda c: -len(c["photos"]))
        for rank, cluster in enumerate(clusters):
            for _, face in cluster["faces"]:
                face["person"] = rank
        for _, face in faces:
            del face["embedding"]

        shown = clusters[:SHEET_PEOPLE]
        columns = min(SHEET_COLUMNS, len(shown))
        sheet = Image.new("RGB", (columns * TILE, math.ceil(len(shown) / columns) * TILE))
        draw = ImageDraw.Draw(sheet)
        font = ImageFont.load_default(size=22)
        for rank, cluster in enumerate(shown):
            i, face = max(cluster["faces"], key=lambda t: face_height(t[1]))
            x, y = rank % columns * TILE, rank // columns * TILE
            tile, to_tile = crop_face(load_rgb(photos[i]["path"]), face["box"])
            tile_draw = ImageDraw.Draw(tile)
            for other in photos[i]["faces"]:
                tile_draw.rectangle(to_tile(other["box"]), outline="red" if other is face else "gray", width=3 if other is face else 2)
            sheet.paste(tile, (x, y))
            label = f"{rank + 1}: {len(cluster['photos'])} photo{'s' * (len(cluster['photos']) != 1)}"
            draw.rectangle(draw.textbbox((x + 4, y + 4), label, font=font), fill="black")
            draw.text((x + 4, y + 4), label, fill="white", font=font)
        return ({"folder": folder, "photos": photos, "counts": [len(c["photos"]) for c in clusters]}, to_tensor(sheet))

    @classmethod
    def IS_CHANGED(cls, folder, min_confidence):
        directory = folder_path(folder)
        return hashlib.sha256(json.dumps([(f, os.path.getmtime(os.path.join(directory, f))) for f in list_images(directory)] + [min_confidence]).encode()).hexdigest()


def parse_corrections(text, photos):
    """Lines like "12: 2" (photo 12 shows the person as its face 2) or "12: none" (they aren't in photo 12), as photo index -> face."""
    corrections = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        number, _, choice = (part.strip().lower() for part in line.partition(":"))
        if not (number.isdigit() and 1 <= int(number) <= len(photos)):
            raise ValueError(f"Select Person: no photo {number!r} in the line {line!r}")
        faces = photos[int(number) - 1]["faces"]
        if choice == "none":
            corrections[int(number) - 1] = None
        elif choice.isdigit() and 1 <= int(choice) <= len(faces):
            corrections[int(number) - 1] = faces[int(choice) - 1]
        else:
            raise ValueError(f"Select Person: photo {number} has {len(faces)} faces, use 1 to {len(faces)} or none, not {choice!r}")
    return corrections


class SelectPerson:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "people": ("PEOPLE", {"tooltip": "Find People's people."}),
                "person": ("INT", {"default": 1, "min": 1, "tooltip": "The number of the person in Find People's preview, 1 is the most frequent."}),
                "corrections": ("STRING", {"multiline": True, "default": "",
                                           "tooltip": "Fixes for photos where the wrong face was picked, one per line, as <photo number>: <face number> "
                                                      "or <photo number>: none when the person isn't in the photo. Both numbers are in Review People's preview, "
                                                      "faces count from the left."}),
            }
        }

    RETURN_TYPES = ("PEOPLE",)
    RETURN_NAMES = ("people",)
    OUTPUT_TOOLTIPS = ("Connect to Review People and Plan Captions.",)
    FUNCTION = "select"
    CATEGORY = "utils/personal"
    DESCRIPTION = "Picks the person in every photo, the corrections override the face recognition."

    def select(self, people, person, corrections):
        if person > len(people["counts"]):
            raise ValueError(f"Select Person: only {len(people['counts'])} people were found")
        fixed = parse_corrections(corrections, people["photos"])
        photos = [{**photo, "subject": fixed[i] if i in fixed else next((f for f in photo["faces"] if f["person"] == person - 1), None)}
                  for i, photo in enumerate(people["photos"])]
        return ({"folder": people["folder"], "photos": photos, "counts": people["counts"]},)


class ReviewPeople:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "people": ("PEOPLE", {"tooltip": "Select Person's people."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("review",)
    OUTPUT_TOOLTIPS = ("Every photo with its number, the picked face in red and the other faces in gray with their numbers.",)
    FUNCTION = "review"
    CATEGORY = "utils/personal"
    DESCRIPTION = "Shows who was picked in every photo, wrong picks go into Select Person's corrections."

    def review(self, people):
        photos = people["photos"]
        columns = min(REVIEW_COLUMNS, len(photos))
        sheet = Image.new("RGB", (columns * REVIEW_TILE, math.ceil(len(photos) / columns) * REVIEW_TILE))
        draw = ImageDraw.Draw(sheet)
        font = ImageFont.load_default(size=18)
        for n, photo in enumerate(photos):
            image = load_rgb(photo["path"])
            image.thumbnail((REVIEW_TILE, REVIEW_TILE))
            x, y = n % columns * REVIEW_TILE, n // columns * REVIEW_TILE
            sheet.paste(image, (x, y))
            draw.rectangle((x, y, x + REVIEW_TILE - 1, y + REVIEW_TILE - 1), outline="#333333")
            for number, face in enumerate(photo["faces"], 1):
                color = "red" if face is photo["subject"] else "gray"
                x0, y0, x1, y1 = (x + face["box"][0] * image.width, y + face["box"][1] * image.height, x + face["box"][2] * image.width, y + face["box"][3] * image.height)
                draw.rectangle((x0, y0, x1, y1), outline=color, width=3 if color == "red" else 2)
                draw.text((x0 + 3, y0 + 1), str(number), fill=color if color == "red" else "white", font=font, stroke_width=2, stroke_fill="black")
            draw.text((x + 4, y + REVIEW_TILE - 24), f"#{n + 1}", fill="yellow", font=font, stroke_width=2, stroke_fill="black")
        return (to_tensor(sheet),)


def people_files():
    directory = output_path(PEOPLE_FOLDER)
    return sorted(f for f in os.listdir(directory) if f.endswith(".json")) if os.path.isdir(directory) else []


class SavePeople:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "people": ("PEOPLE", {"tooltip": "Select Person's people."}),
                "name": ("STRING", {"default": "", "tooltip": "The file name in output/people, empty for the name of the photo folder."}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file",)
    OUTPUT_TOOLTIPS = ("The path of the saved file.",)
    OUTPUT_NODE = True
    FUNCTION = "save"
    CATEGORY = "utils/personal"
    DESCRIPTION = "Saves who is in every photo to output/people, Load People reads it in another workflow."

    def save(self, people, name):
        path = output_path(f"{PEOPLE_FOLDER}/{name.strip() or os.path.basename(people['folder'])}.json")
        photos = [{"name": p["name"], "faces": [f["box"] for f in p["faces"]], "subject": next((i for i, f in enumerate(p["faces"]) if f is p["subject"]), None)}
                  for p in people["photos"]]
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"folder": people["folder"], "photos": photos}, f)
        return (path,)


class LoadPeople:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file": (people_files(), {"tooltip": "A file Save People wrote to output/people, press R to refresh the list."}),
            }
        }

    RETURN_TYPES = ("PEOPLE",)
    RETURN_NAMES = ("people",)
    OUTPUT_TOOLTIPS = ("Connect to Plan Captions.",)
    FUNCTION = "load"
    CATEGORY = "utils/personal"
    DESCRIPTION = "Loads the photos and who is in them from a file Save People wrote."

    def load(self, file):
        with open(output_path(f"{PEOPLE_FOLDER}/{file}"), encoding="utf-8") as f:
            saved = json.load(f)
        photos = []
        for p in saved["photos"]:
            faces = [{"box": box} for box in p["faces"]]
            photos.append({"name": p["name"], "path": os.path.join(saved["folder"], p["name"]), "faces": faces,
                           "subject": None if p["subject"] is None else faces[p["subject"]]})
        return ({"folder": saved["folder"], "photos": photos},)

    @classmethod
    def IS_CHANGED(cls, file):
        return os.path.getmtime(output_path(f"{PEOPLE_FOLDER}/{file}"))


class PlanCaptions:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "people": ("PEOPLE", {"tooltip": "Load People's people."}),
                "trigger": ("STRING", {"default": "", "tooltip": "The trigger word that replaces <trigger> in the prompt."}),
                "sex": ("STRING", {"default": "woman", "tooltip": "Replaces <sex> in the prompt, e.g. woman, man or girl."}),
                "prompt": ("STRING", {"multiline": True, "default": "the trigger is <trigger>, the <sex> <position>",
                                      "tooltip": "The opening of every caption. <position> becomes where the person is among the others "
                                                 "(on the left, on the right, in the middle, in the front or in the back) and nothing in photos where they are alone."}),
                "skip_captioned": ("BOOLEAN", {"default": False, "tooltip": "Leave out the photos that already have a .txt file next to them."}),
            }
        }

    RETURN_TYPES = ("JSON_LIST", "INT", "STRING")
    RETURN_NAMES = ("items", "count", "report")
    OUTPUT_TOOLTIPS = ("One item per photo, connect it to ▶ For Each.", "The number of photos to caption.",
                       "How many photos the person is in, connect to Preview as Text.")
    FUNCTION = "plan"
    CATEGORY = "utils/personal"
    DESCRIPTION = "Plans the caption opening of every photo."

    def plan(self, people, trigger, sex, prompt, skip_captioned):
        items = []
        found = faceless = 0
        for photo in people["photos"]:
            face = photo["subject"]
            found += face is not None
            faceless += not photo["faces"]
            if skip_captioned and os.path.exists(caption_path(photo["path"])):
                continue
            where = position(face, [f for f in photo["faces"] if f is not face])
            opening = prompt.replace("<trigger>", trigger).replace("<sex>", sex).replace("<position>", where)
            items.append({"name": photo["name"], "path": photo["path"], "prompt": " ".join(opening.split()).rstrip(" ,"),
                          "box": face["box"] if face else None})
        report = f"The person is in {found} of {len(people['photos'])} photos, {faceless} have no face, {len(items)} to caption"
        return (items, len(items), report)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")  # skip_captioned depends on the files next to the photos


class LoadItemImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "item": (ANY, {"tooltip": "▶ For Each's item from Plan Captions."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("image", "marked")
    OUTPUT_TOOLTIPS = ("The photo.", "The photo with a red box around the person's face, the same photo when they weren't found.")
    FUNCTION = "load"
    CATEGORY = "utils/personal"
    DESCRIPTION = "Loads the photo of the current item and marks the person on a copy."

    def load(self, item):
        photo = load_rgb(item["path"])
        marked = photo.copy()
        if item["box"]:
            x0, y0, x1, y1 = item["box"]
            ImageDraw.Draw(marked).rectangle((x0 * photo.width, y0 * photo.height, x1 * photo.width, y1 * photo.height),
                                             outline="red", width=max(3, round(photo.height / 200)))
        return (to_tensor(photo), to_tensor(marked))


class SaveCaption:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "item": (ANY, {"tooltip": "▶ For Each's item from Plan Captions."}),
                "text": ("STRING", {"forceInput": True, "tooltip": "The caption."}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file",)
    OUTPUT_TOOLTIPS = ("The path of the .txt file, connect to For Each ◀.",)
    FUNCTION = "save"
    CATEGORY = "utils/personal"
    DESCRIPTION = "Saves the caption as a .txt file with the photo's name next to the photo."

    def save(self, item, text):
        path = caption_path(item["path"])
        with open(path, "w", encoding="utf-8") as f:
            f.write(text.strip())
        return (path,)
