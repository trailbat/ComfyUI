# Reference Analyzer scores every photo in an input subfolder against every prompt of a prompt list, Reference Picker
# outputs the photos picked for the For Each loop's current prompt. js/reference_picker.js shows the scores as a grid
# on the Picker, where the picks and the subject's anchor photos are chosen.
#
# Signals: InsightFace (subject identity, head yaw/pitch, eyes, face size), YOLO pose (crop level, posture, rear views),
# the YOLO hand detector, CLIP ViT-L/14 (lighting, environment, camera angle, posture, hair) and plain image statistics
# (sharpness, exposure, color cast). Components are computed as 0-1 and sent as 0-100, the JS weighs them into one score.

import hashlib
import json
import math
import os

import cv2
import numpy as np
import torch
from insightface.app import FaceAnalysis
from PIL import Image, ImageOps
from ultralytics import YOLO

import comfy.clip_vision
import comfy.model_management
import comfy.sd
import comfy.utils
import folder_paths
import node_helpers

ANY = "*"

INSIGHTFACE_DIR = os.path.join(folder_paths.models_dir, "insightface", "models", "buffalo_l")
POSE_MODEL = os.path.join(folder_paths.models_dir, "ultralytics", "pose", "yolo11m-pose.pt")
HAND_MODEL = os.path.join(folder_paths.models_dir, "ultralytics", "bbox", "hand_yolov8s.pt")

COMPONENTS = ("angle", "crop", "pose", "camera", "gaze", "light", "environment", "quality", "identity")
MAX_PICKS = 8
# Cosine similarity to the anchors above which a face counts as the subject (w600k_r50 embeddings).
SUBJECT_THRESHOLD = 0.35
DETECT_SIZE = 2048  # longest side the detectors see

# COCO keypoints
NOSE, L_EYE, R_EYE = 0, 1, 2
L_SHOULDER, R_SHOULDER, L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE = 5, 6, 11, 12, 13, 14, 15, 16

CROP_LEVELS = {"head_shoulders": 1, "medium": 2, "cowboy": 3, "middle_full": 3, "full": 4, "wide": 4}
ANGLE_YAW = {"front": 0, "three_quarter": 35, "profile": 70}
CAMERA_TEXT = {"high": "a high-angle photo looking down at a person",
               "eye_level": "an eye-level photo of a person",
               "low": "a low-angle photo looking up at a person"}
POSTURE_TEXT = {"standing": "a photo of a person standing", "seated": "a photo of a person sitting",
                "walking": "a photo of a person walking", "jumping": "a photo of a person jumping in the air"}
# Neighbors on this scale (blonde and light brown, ...) are too close for CLIP to tell apart reliably.
HAIR_SHADES = ("blonde", "light brown", "dark brown", "black")
HAIR_COLORS = HAIR_SHADES + ("red", "grey")
HAIR_LENGTHS = ("short", "shoulder-length", "long")


def input_path(relative):
    """Absolute path of a file or folder inside the input directory, refusing anything that escapes it."""
    base = os.path.abspath(folder_paths.get_input_directory())
    path = os.path.abspath(os.path.join(base, relative))
    if os.path.commonpath((base, path)) != base:
        raise ValueError(f"Not inside the input folder: {relative!r}")
    return path


def list_photos(folder):
    directory = input_path(folder)
    if not os.path.isdir(directory):
        raise ValueError(f"Reference Analyzer: input/{folder} doesn't exist, upload photos with the Reference Picker first")
    files = sorted(f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)))
    return [f"{folder}/{f}" if folder else f for f in folder_paths.filter_files_content_types(files, ["image"])]


def load_photo(key):
    image = node_helpers.pillow(Image.open, input_path(key))
    return node_helpers.pillow(ImageOps.exif_transpose, image).convert("RGB")


def hair_differs(a, b):
    if a in HAIR_SHADES and b in HAIR_SHADES:
        return abs(HAIR_SHADES.index(a) - HAIR_SHADES.index(b)) > 1
    return a != b


def gauss(x, center, width):
    return math.exp(-((x - center) / width) ** 2)


def clamp(x):
    return min(1.0, max(0.0, x))


def normalize(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def softmax(logits):
    e = np.exp(logits - logits.max())
    return e / e.sum()


def laplacian_var(gray, height):
    scale = height / gray.shape[0]
    gray = cv2.resize(gray, (max(1, round(gray.shape[1] * scale)), height), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def pupil_offset(gray, landmarks, eye):
    """Horizontal pupil position inside one eye, -0.5 at the image-left corner to 0.5 at the image-right corner: the
    centroid of the darkest fifth of the pixels inside the eye outline (68-point landmarks 36-41 or 42-47)."""
    points = landmarks[eye:eye + 6, :2]
    x0, y0 = np.floor(points.min(0)).astype(int)
    x1, y1 = np.ceil(points.max(0)).astype(int)
    x0, y0 = max(x0, 0), max(y0, 0)
    if x1 - x0 < 6 or y1 - y0 < 2:
        return None
    crop = gray[y0:y1 + 1, x0:x1 + 1]
    mask = np.zeros(crop.shape, np.uint8)
    cv2.fillPoly(mask, [np.round(points - (x0, y0)).astype(np.int32)], 1)
    values = crop[mask > 0]
    if values.size < 8:
        return None
    dark = (crop <= np.percentile(values, 20)) & (mask > 0)
    cx = np.nonzero(dark)[1].mean() + x0
    left, right = points[:, 0].min(), points[:, 0].max()
    return float((cx - left) / (right - left) - 0.5)


def eye_openness(landmarks):
    """Eye aspect ratio averaged over both eyes, about 0.3 open and under 0.15 closed."""
    ratios = []
    for eye in (36, 42):
        p = landmarks[eye:eye + 6, :2]
        ratios.append((np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])) / (2 * np.linalg.norm(p[0] - p[3]) + 1e-6))
    return float(np.mean(ratios))


def subject_embedding(faces_per_photo):
    """Mean embedding over photos, each photo contributing its largest face first and then the face closest to that
    first mean, so a stranger in one photo doesn't pull the result."""
    picks = [max(faces, key=lambda f: f["area"]) for faces in faces_per_photo if faces]
    if not picks:
        return None
    mean = normalize(np.mean([f["embedding"] for f in picks], 0))
    picks = [max(faces, key=lambda f: float(f["embedding"] @ mean)) for faces in faces_per_photo if faces]
    return normalize(np.mean([f["embedding"] for f in picks], 0))


class Detectors:
    def __init__(self):
        for path in (INSIGHTFACE_DIR, POSE_MODEL, HAND_MODEL):
            if not os.path.exists(path):
                raise FileNotFoundError(f"Reference Analyzer needs {path}")
        self.faces = FaceAnalysis(name=INSIGHTFACE_DIR, allowed_modules=["detection", "recognition", "landmark_3d_68"])
        self.faces.prepare(ctx_id=0, det_size=(640, 640))
        self.pose = YOLO(POSE_MODEL)
        self.hands = YOLO(HAND_MODEL)
        self.device = str(comfy.model_management.get_torch_device())

    def detect(self, photo):
        """Everything about one photo that doesn't depend on the subject or the prompts."""
        scale = min(1.0, DETECT_SIZE / max(photo.size))
        if scale < 1:
            photo = photo.resize((round(photo.width * scale), round(photo.height * scale)), Image.LANCZOS)
        rgb = np.asarray(photo)
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])
        faces = [{"bbox": f.bbox.astype(float), "score": float(f.det_score), "embedding": normalize(f.embedding.astype(np.float32)),
                  "pitch": float(f.pose[0]), "yaw": float(f.pose[1]), "landmarks": f.landmark_3d_68,
                  "area": float((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))} for f in self.faces.get(bgr)]
        pose = self.pose.predict(bgr, device=self.device, verbose=False)[0]
        people = []
        if pose.keypoints is not None and pose.keypoints.conf is not None:
            people = [{"box": box, "xy": xy, "conf": conf} for box, xy, conf in
                      zip(pose.boxes.xyxy.cpu().numpy(), pose.keypoints.xy.cpu().numpy(), pose.keypoints.conf.cpu().numpy())]
        hands = self.hands.predict(bgr, device=self.device, verbose=False, conf=0.4)[0].boxes.xyxy.cpu().numpy()
        return {"rgb": rgb, "scale": scale, "faces": faces, "people": people, "hands": hands}


class ClipScorer:
    def __init__(self, clip, vision):
        self.clip = clip
        self.vision = vision
        self.texts = {}

    def text(self, text):
        if text not in self.texts:
            pooled = self.clip.encode_from_tokens(self.clip.tokenize(text), return_pooled=True)[1]
            self.texts[text] = normalize(pooled[0].float().cpu().numpy())
        return self.texts[text]

    def image(self, rgb):
        pixels = torch.from_numpy(rgb.astype(np.float32) / 255.0)[None]
        return normalize(self.vision.encode_image(pixels)["image_embeds"][0].float().cpu().numpy())

    def probabilities(self, embedding, texts):
        return softmax(100 * np.array([float(embedding @ self.text(t)) for t in texts]))


def subject_person(people, face):
    """The pose detection belonging to the subject: the one whose head keypoints are inside the subject's face box,
    or the largest person when there is no face."""
    if not people:
        return None
    if face is None:
        return max(people, key=lambda p: (p["box"][2] - p["box"][0]) * (p["box"][3] - p["box"][1]))
    x0, y0, x1, y1 = face["bbox"]
    for person in people:
        for k in (NOSE, L_EYE, R_EYE):
            x, y = person["xy"][k]
            if person["conf"][k] > 0.3 and x0 <= x <= x1 and y0 <= y <= y1:
                return person
    return None


def crop_level(person, height):
    """0 face only, 1 head and shoulders, 2 down to the hips, 3 down to the knees, 4 full body."""
    visible = lambda *ks: any(person["conf"][k] > 0.5 and person["xy"][k][1] < height * 0.98 for k in ks)
    if visible(L_ANKLE, R_ANKLE):
        return 4
    if visible(L_KNEE, R_KNEE):
        return 3
    if visible(L_HIP, R_HIP):
        return 2
    if visible(L_SHOULDER, R_SHOULDER):
        return 1
    return 0


def leg_posture(person):
    """Standing / seated / walking probabilities from the keypoints, None when the legs aren't visible."""
    xy, conf = person["xy"], person["conf"]
    if min(conf[[L_SHOULDER, R_SHOULDER, L_HIP, R_HIP, L_KNEE, R_KNEE]]) < 0.5:
        return None
    shoulder, hip, knee = (xy[L_SHOULDER] + xy[R_SHOULDER]) / 2, (xy[L_HIP] + xy[R_HIP]) / 2, (xy[L_KNEE] + xy[R_KNEE]) / 2
    torso = np.linalg.norm(shoulder - hip) + 1e-6
    thigh_drop = (knee[1] - hip[1]) / torso  # about 0.9 standing, near 0 seated
    seated = clamp((0.6 - thigh_drop) / 0.4)
    walking = 0.0
    if min(conf[[L_ANKLE, R_ANKLE]]) > 0.5:
        # Kneeling and cross-legged keep the thighs upright but fold the shins, count them as seated.
        shin_drop = ((xy[L_ANKLE][1] + xy[R_ANKLE][1]) / 2 - knee[1]) / torso
        seated = max(seated, clamp((0.6 - shin_drop) / 0.4))
        walking = clamp((abs(xy[L_ANKLE][0] - xy[R_ANKLE][0]) / torso - 0.25) / 0.35) * (1 - seated)
    return {"standing": (1 - seated) * (1 - walking), "seated": seated, "walking": walking}


def describe(key, detection, subject, clip):
    """Per-photo features for scoring, the subject's face and body picked out with the subject embedding."""
    rgb, scale = detection["rgb"], detection["scale"]
    height, width = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    faces = detection["faces"]
    notes = []

    face = None
    match = None
    if faces and subject is not None:
        best = max(faces, key=lambda f: float(f["embedding"] @ subject))
        match = float(best["embedding"] @ subject)
        if match >= SUBJECT_THRESHOLD:
            face = best
    found = face is not None
    if faces and not found:
        notes.append(f"subject not found (best match {match:.2f})")
    elif not faces:
        notes.append("no face, identity not checked")
    else:
        notes.append(f"match {match:.2f}")

    others = sum(1 for f in faces if f is not face and f["score"] > 0.6 and (f["bbox"][3] - f["bbox"][1]) > 0.03 * height)
    if others:
        notes.append(f"{others} other face{'s' if others > 1 else ''}")

    info = {"key": key, "found": found, "has_face": bool(faces), "others": others, "match": match, "notes": notes,
            "megapixels": rgb.shape[0] * rgb.shape[1] / scale ** 2 / 1e6, "box": None}
    embedding = clip.image(rgb)
    info["clip"] = embedding
    info["camera"] = dict(zip(CAMERA_TEXT, clip.probabilities(embedding, list(CAMERA_TEXT.values()))))
    info["posture_clip"] = dict(zip(POSTURE_TEXT, clip.probabilities(embedding, list(POSTURE_TEXT.values()))))

    means = rgb.reshape(-1, 3).mean(0)
    info["color_cast"] = float(means.max() / max(means.min(), 1.0)) > 2.2  # warm indoor light reaches about 2
    if info["color_cast"]:
        notes.append("strong color cast")
    info["sharp"] = clamp(laplacian_var(gray, 1024) / 200)

    if face is not None:
        x0, y0, x1, y1 = face["bbox"]
        info["box"] = [round(float(v), 3) for v in (x0 / width, y0 / height, x1 / width, y1 / height)]
        info["face_px"] = (y1 - y0) / scale
        info["face_frac"] = (y1 - y0) / height
        info["turn"] = face["yaw"]  # positive: the face points toward the right edge of the frame
        info["pitch"] = face["pitch"]
        landmarks = face["landmarks"]
        offsets = [o for o in (pupil_offset(gray, landmarks, 36), pupil_offset(gray, landmarks, 42)) if o is not None]
        info["eye_turn"] = float(np.mean(offsets)) * 90 if offsets else 0.0
        info["openness"] = eye_openness(landmarks)
        fx0, fy0, fx1, fy1 = max(0, int(x0)), max(0, int(y0)), min(width, int(x1)), min(height, int(y1))
        face_gray = gray[fy0:fy1, fx0:fx1]
        sharpness = laplacian_var(face_gray, 256)
        info["face_sharp"] = clamp(sharpness / 200)
        if sharpness < 45:
            notes.append("blurry face")
        half = face_gray.shape[1] // 2
        left, right = face_gray[:, :half].mean(), face_gray[:, half:].mean()
        info["light_asym"] = float((left - right) / (left + right + 1e-6))  # positive: lit from the frame's left
        info["blown"] = float((face_gray > 250).mean())
        if info["blown"] > 0.15:
            notes.append("blown highlights on face")
        # The head with hair: the face box widened and extended up and down.
        fw, fh = x1 - x0, y1 - y0
        head = rgb[max(0, int(y0 - 0.8 * fh)):min(height, int(y1 + 1.2 * fh)), max(0, int(x0 - fw)):min(width, int(x1 + fw))]
        head_embedding = clip.image(head)
        info["hair_color"] = HAIR_COLORS[int(np.argmax(clip.probabilities(head_embedding, [f"a photo of a person with {c} hair" for c in HAIR_COLORS])))]
        info["hair_length"] = HAIR_LENGTHS[int(np.argmax(clip.probabilities(head_embedding, [f"a photo of a person with {c} hair" for c in HAIR_LENGTHS])))]

    person = subject_person(detection["people"], face)
    if faces and face is None:
        person = None  # only strangers in the photo
    info["level"] = crop_level(person, height) if person is not None else (0 if found and info["face_frac"] > 0.3 else 1 if found else None)
    info["person_frac"] = float((person["box"][3] - person["box"][1]) / height) if person is not None else None
    info["person_px"] = float((person["box"][3] - person["box"][1]) / scale) if person is not None else 0.0
    info["legs"] = leg_posture(person) if person is not None else None
    info["ankles"] = int(sum(person["conf"][k] > 0.5 for k in (L_ANKLE, R_ANKLE))) if person is not None else 0
    info["rear"] = person is not None and not faces and min(person["conf"][[L_SHOULDER, R_SHOULDER]]) > 0.5

    hands = detection["hands"]
    if person is not None and len(hands):
        bx0, by0, bx1, by1 = person["box"]
        centers = (hands[:, :2] + hands[:, 2:]) / 2
        hands = hands[(centers[:, 0] >= bx0) & (centers[:, 0] <= bx1) & (centers[:, 1] >= by0) & (centers[:, 1] <= by1)]
    info["hand_count"] = len(hands)
    info["hand_px"] = float((hands[:, 3] - hands[:, 1]).max() / scale) if len(hands) else 0.0
    return info


def score(info, target, clip, light_rank, environment_rank):
    """Component scores 0-1 for one photo and prompt, in COMPONENTS order, plus the penalty."""
    found = info["found"]
    crop = target.get("crop", "head_shoulders")

    angle = target.get("angle", "any")
    if angle == "any":
        angle_score = 1.0
    elif angle == "rear":
        angle_score = 1.0 if info["rear"] else 0.6 if found and abs(info["turn"]) > 60 else 0.0
    elif found:
        angle_score = gauss(abs(info["turn"]), ANGLE_YAW[angle], 15 if angle != "profile" else 20)
        side = target.get("side")
        if side and abs(info["turn"]) > 10 and (info["turn"] > 0) != (side == "right"):
            angle_score *= 0.75  # the other side still helps, faces are close to symmetric
    else:
        angle_score = 0.0

    level = info["level"]
    if crop in ("eyes", "face_detail", "ear"):
        crop_score = clamp(info["face_px"] / 600) if found else 0.0
    elif crop == "hands":
        crop_score = clamp(info["hand_px"] / 250) * (1.0 if info["hand_count"] >= 2 else 0.6) if info["hand_count"] else 0.0
    elif crop == "hand_face":
        crop_score = (0.5 * clamp(info["face_px"] / 300) if found else 0.0) + (0.5 if info["hand_count"] else 0.0)
    elif crop == "feet":
        crop_score = info["ankles"] / 2 * clamp(info["person_px"] / 1200)
    elif level is None:
        crop_score = 0.0
    else:
        crop_score = max(0.0, 1 - 0.4 * abs(level - CROP_LEVELS[crop]))
        if crop == "head_shoulders" and level == 0:
            crop_score = 0.8
        if crop == "wide" and (info["person_frac"] or 1) > 0.45:
            crop_score *= 0.5

    posture = target.get("posture", "any")
    if posture == "any":
        pose_score = 1.0
    elif info["legs"] is not None and posture in info["legs"]:
        pose_score = 0.5 * info["legs"][posture] + 0.5 * info["posture_clip"][posture]
    else:
        pose_score = 0.5 + 0.5 * info["posture_clip"][posture]  # legs out of frame, CLIP alone is a weak signal

    camera = target.get("camera", "eye_level")
    camera_score = info["camera"][camera]
    if found:
        tilt = info["tilt"]  # positive: chin up, the camera is above
        head = clamp(0.5 + tilt / 30) if camera == "high" else clamp(0.5 - tilt / 30) if camera == "low" else gauss(tilt, 0, 15)
        camera_score = 0.5 * camera_score + 0.5 * head

    gaze = target.get("gaze", "any")
    if gaze == "any":
        gaze_score = 1.0
    elif not found:
        gaze_score = 0.3
    else:
        opened = clamp((info["openness"] - 0.12) / 0.1)
        at_camera = gauss(info["turn"] + info["eye_turn"], 0, 15) * gauss(info["tilt"], 0, 25) * opened
        gaze_score = {"camera": at_camera, "away": (1 - at_camera) * opened, "down": clamp(-info["tilt"] / 25) * 0.5 + 0.5 * (1 - at_camera),
                      "closed": 1 - opened}[gaze]

    light_score = light_rank
    side = target.get("light_side", "any")
    if found and side in ("left", "right", "even"):
        asym = info["light_asym"]
        direction = clamp(0.5 + asym * 5) if side == "left" else clamp(0.5 - asym * 5) if side == "right" else clamp(1 - abs(asym) * 4)
        light_score = 0.7 * light_rank + 0.3 * direction

    if found:
        quality = 0.6 * info["face_sharp"] + 0.2 * clamp(info["megapixels"] / 3) + 0.2 * clamp(info["face_px"] / 400)
        quality *= 1 - info["blown"]
        identity = clamp((info["match"] - 0.4) / 0.5)
    else:
        quality = 0.7 * info["sharp"] + 0.3 * clamp(info["megapixels"] / 3)
        identity = 0.3 if not info["has_face"] else 0.0

    penalty = min(0.2, 0.07 * info["others"]) + (0.1 if info.get("hair_outlier") else 0.0) + (0.05 if info["color_cast"] else 0.0)
    return [round(100 * v) for v in (angle_score, crop_score, pose_score, camera_score, gaze_score, light_score,
                                     environment_rank, quality, identity, penalty)]


def rank_similarity(values):
    """Raw CLIP cosines sit in a narrow band, spread them over 0-1 relative to the other photos."""
    values = np.array(values)
    if len(values) < 2 or values.std() < 1e-6:
        return np.full(len(values), 0.5)
    return 1 / (1 + np.exp(-1.5 * (values - values.mean()) / values.std()))


class ReferenceAnalyzer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip_model": (folder_paths.get_filename_list("clip_vision"),
                               {"tooltip": "OpenAI CLIP ViT-L/14 with both halves (openai/clip-vit-large-patch14 model.safetensors), "
                                           "the text and image side must come from the same model."}),
                "prompts": ("JSON_LIST", {"tooltip": "Load Prompt List's prompts, see lora_dataset_prompts.yaml for the targets each prompt can set."}),
                "folder": ("STRING", {"default": "lora_refs", "tooltip": "The subfolder of input with the photos, the Reference Picker uploads to it."}),
                "anchors": ("STRING", {"multiline": True, "default": "",
                                       "tooltip": "Photos that clearly show the subject, one per line. Star them in the Reference Picker. "
                                                  "Empty: the subject is the face that appears most across the photos."}),
            }
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "analyze"
    CATEGORY = "utils/personal"
    DESCRIPTION = "Scores every photo in the folder against every prompt, the Reference Picker shows the result."

    def analyze(self, clip_model, prompts, folder, anchors):
        folder = folder.strip().strip("/\\")
        keys = list_photos(folder)
        if not keys:
            raise ValueError(f"Reference Analyzer: no photos in input/{folder}")
        anchor_keys = [a.strip() for a in anchors.splitlines() if a.strip() in keys]

        path = folder_paths.get_full_path_or_raise("clip_vision", clip_model)
        clip = ClipScorer(comfy.sd.load_clip([path], embedding_directory=folder_paths.get_folder_paths("embeddings")), comfy.clip_vision.load(path))
        detectors = Detectors()
        progress = comfy.utils.ProgressBar(len(keys) * 2)
        detections = {}
        for key in keys:
            detections[key] = detectors.detect(load_photo(key))
            progress.update(1)
        del detectors

        if anchor_keys:
            subject = subject_embedding([detections[k]["faces"] for k in anchor_keys])
            if subject is None:
                raise ValueError("Reference Analyzer: none of the anchor photos has a face that can be detected")
        else:
            subject = subject_embedding([d["faces"] for d in detections.values()])

        photos = []
        for key in keys:
            photos.append(describe(key, detections.pop(key), subject, clip))
            progress.update(1)

        found = [p for p in photos if p["found"]]
        if found:
            median_pitch = float(np.median([p["pitch"] for p in found]))  # head pitch has a per-detector bias
            for p in found:
                p["tilt"] = p["pitch"] - median_pitch
            reference = [p for p in found if p["key"] in anchor_keys] or found
            label = "anchors" if anchor_keys else "most photos"
            hair = max(HAIR_COLORS, key=lambda c: sum(p["hair_color"] == c for p in reference))
            length = max(HAIR_LENGTHS, key=lambda c: sum(p["hair_length"] == c for p in reference))
            for p in found:
                if hair_differs(p["hair_color"], hair):
                    p["hair_outlier"] = True
                    p["notes"].append(f"{p['hair_color']} hair, {label} {hair}")
                elif p["hair_length"] != length:
                    p["notes"].append(f"{p['hair_length']} hair, {label} {length}")

        embeddings = np.stack([p["clip"] for p in photos])
        similarity = embeddings @ embeddings.T
        result_prompts = []
        for i, target in enumerate(prompts):
            pid = str(target.get("id", i + 1))
            light = rank_similarity(embeddings @ clip.text(f"a photo with {target['light']}")) if target.get("light") else np.full(len(photos), 0.5)
            environment = rank_similarity(embeddings @ clip.text(f"a photo of {target['environment']}")) if target.get("environment") else np.full(len(photos), 0.5)
            result_prompts.append({
                "id": pid,
                "title": str(target.get("title", target.get("prompt", ""))[:80]),
                "tags": " · ".join(str(target[k]) for k in ("crop", "angle", "side", "camera", "gaze", "posture") if target.get(k) not in (None, "any")),
                "prompt": str(target.get("prompt", "")),
                "scores": [score(p, target, clip, float(light[j]), float(environment[j])) for j, p in enumerate(photos)],
            })

        # Scores are 0-100, one list per photo in the order of "photos", the last value is the penalty.
        analysis = {
            "folder": folder,
            "anchors": anchor_keys,
            "components": COMPONENTS,
            "photos": [{"key": p["key"], "found": p["found"] or not p["has_face"], "box": p["box"], "notes": p["notes"],
                        "dupes": [int(j) for j in np.nonzero(similarity[i] > 0.97)[0] if j != i]}
                       for i, p in enumerate(photos)],
            "prompts": result_prompts,
        }
        return {"ui": {"analysis": [analysis]}}

    @classmethod
    def IS_CHANGED(cls, folder, **kwargs):
        folder = folder.strip().strip("/\\")
        return hashlib.sha256(json.dumps([(k, os.path.getmtime(input_path(k))) for k in list_photos(folder)]).encode()).hexdigest()


def parse_picks(selection):
    return json.loads(selection).get("picks", {}) if selection.strip() else {}


class ReferencePicker:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "item": (ANY, {"tooltip": "▶ For Each's item, a prompt from the prompt list with an id."}),
                "selection": ("STRING", {"default": "", "tooltip": "Written by the picker grid: the analysis and the picked photos per prompt."}),
            }
        }

    RETURN_TYPES = ("IMAGE",) * MAX_PICKS
    RETURN_NAMES = tuple(f"image_{i}" for i in range(1, MAX_PICKS + 1))
    OUTPUT_TOOLTIPS = ("The picked photos in rank order, empty slots output nothing and are skipped by encoders "
                       "such as Text Encode Qwen Image 2.1.",) * MAX_PICKS
    FUNCTION = "pick"
    CATEGORY = "utils/personal"
    DESCRIPTION = "Outputs the reference photos picked for the current prompt, pick them in the grid after running the Reference Analyzer."

    def pick(self, item, selection):
        keys = parse_picks(selection).get(str(item.get("id", "")), [])[:MAX_PICKS]
        dtype = comfy.model_management.intermediate_dtype()
        device = comfy.model_management.intermediate_device()
        images = [torch.from_numpy(np.asarray(load_photo(k), dtype=np.float32) / 255.0)[None].to(device=device, dtype=dtype) for k in keys]
        return tuple(images + [None] * (MAX_PICKS - len(images)))

    @classmethod
    def IS_CHANGED(cls, selection, **kwargs):
        keys = sorted({k for picks in parse_picks(selection).values() for k in picks})
        return hashlib.sha256(json.dumps([(k, os.path.getmtime(input_path(k))) for k in keys]).encode()).hexdigest()
