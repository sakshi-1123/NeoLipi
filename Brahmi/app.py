import os
import csv
import glob
import json
import tempfile
import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image


# ============================================================
# STREAMLIT DRAWABLE CANVAS COMPATIBILITY
# ============================================================
# streamlit-drawable-canvas 0.9.3 imports:
#
#     streamlit.elements.image.image_to_url
#
# Newer Streamlit versions moved image_to_url to:
#
#     streamlit.elements.lib.image_utils
#
# Without this compatibility shim the manual segmentation
# editor crashes when it tries to display the background image.
try:
    import streamlit.elements.image as _st_image

    # streamlit-drawable-canvas 0.9.3 calls the OLD Streamlit
    # image_to_url API positionally:
    #
    #   image_to_url(image, width, clamp, channels, output_format, image_id)
    #
    # New Streamlit versions moved image_to_url and changed its
    # second argument to a LayoutConfig object.  Simply aliasing
    # the new function is therefore NOT enough: the old integer
    # width gets passed as layout_config, producing:
    #   AttributeError: 'int' object has no attribute 'width'
    #
    # This wrapper converts the old call signature to the new API.
    from streamlit.elements.lib import image_utils as _image_utils
    from streamlit.elements.lib.layout_utils import LayoutConfig

    _new_image_to_url = _image_utils.image_to_url

    def _drawable_canvas_image_to_url(
        image,
        width=None,
        clamp=False,
        channels="RGB",
        output_format="auto",
        image_id=None,
        *args,
        **kwargs,
    ):
        # Support both the old positional API used by
        # streamlit-drawable-canvas and a possible keyword call.
        if "layout_config" in kwargs:
            return _new_image_to_url(
                image,
                layout_config=kwargs["layout_config"],
                clamp=clamp,
                channels=channels,
                output_format=output_format,
                image_id=image_id,
            )

        layout_config = LayoutConfig(width=width)

        return _new_image_to_url(
            image,
            layout_config=layout_config,
            clamp=clamp,
            channels=channels,
            output_format=output_format,
            image_id=image_id,
        )

    _st_image.image_to_url = _drawable_canvas_image_to_url

except Exception:
    # The actual canvas import below will provide the useful
    # error/warning if the installed Streamlit version differs.
    pass


try:
    from streamlit_drawable_canvas import st_canvas
    DRAWABLE_CANVAS_AVAILABLE = True
except ImportError:
    st_canvas = None
    DRAWABLE_CANVAS_AVAILABLE = False


# ============================================================
# NeoLipi - One Click Brahmi Inscription Analyzer
# ============================================================
#
# USER:
#   Upload image -> Analyze
#
# PIPELINE:
#   1. Script classification
#   2. Binarization
#   3. Line/character segmentation
#   4. Character extraction
#   5. Save extracted characters
#   6. Preprocess complete extracted folder
#   7. All-class CNN recognition
#   8. Brahmi class -> Unicode/glyph mapping
#   7. Digital Brahmi text
#   8. CSV class-name -> Kannada mapping
#
# Kannada conversion is performed directly from the recognized class names.
#
# The mapping source is the user's CSV:
#   brahmi_mapping.csv
#
# No technical file paths are shown in the UI.
# ============================================================


st.set_page_config(
    page_title="NeoLipi",
    page_icon="🪨",
    layout="centered",
)

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)

# ============================================================
# MANUALLY EXTRACTED CHARACTER OUTPUT
# ============================================================
# Character crops created through the manual segmentation editor
# are saved separately from the automatic "characters" folder.
#
# Output:
#   segmentation_brahmi_v6/
#       manual_characters/
#           line_01_char_001.png
#           line_01_char_002.png
#           ...
#
# Existing binarization, segmentation, CNN, dataset similarity,
# and decision-engine behavior are not changed by this folder.
MANUAL_CHARACTERS_DIR = (
    ROOT / "segmentation_brahmi_v6" / "manual_characters"
)
MANUAL_CHARACTERS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ============================================================
# BRAHMI RECOGNIZER DATASET
# ============================================================
# The dataset shown in the project screenshot is organized as:
#
#   RecognizerDataset_150_210/
#       a/
#       ba/
#       ca/
#       va/
#       ...
#
# Each folder name is the class label.
BRAHMI_DATASET = Path(
    r"D:\NeoLipi_Dataset\RecognizerDataset_150_210\RecognizerDataset_150_210"
)


# ============================================================
# DISCOVERY
# ============================================================

def find_first(paths):
    for path in paths:
        if path.exists():
            return path
    return None


def find_project_file(*names):
    for name in names:
        path = ROOT / name
        if path.exists():
            return path

    # Search recursively, because the CSV may be in a dataset folder.
    for name in names:
        matches = list(ROOT.rglob(name))
        if matches:
            return matches[0]

    return None


def find_script_model():
    exact = find_first([
        MODELS / "script_classifier.keras",
        MODELS / "script_classifier.h5",
        MODELS / "script_classifier_model.keras",
        MODELS / "script_classifier_model.h5",
    ])
    if exact:
        return exact

    files = []
    for pattern in (
        MODELS / "*script*.keras",
        MODELS / "*script*.h5",
        MODELS / "*classifier*.keras",
        MODELS / "*classifier*.h5",
    ):
        files.extend(glob.glob(str(pattern)))

    return Path(sorted(set(files))[0]) if files else None


def find_brahmi_model():
    exact = find_first([
        MODELS / "best_brahmi_final_v2.keras",
        MODELS / "best_brahmi_final_v2.h5",
        MODELS / "brahmi_model.keras",
        MODELS / "brahmi_model.h5",
    ])
    if exact:
        return exact

    files = []
    for pattern in (
        MODELS / "*brahmi*.keras",
        MODELS / "*brahmi*.h5",
    ):
        files.extend(glob.glob(str(pattern)))

    return Path(sorted(set(files))[0]) if files else None


def find_brahmi_classes():
    return find_first([
        MODELS / "brahmi_classes.json",
        MODELS / "classes.json",
    ])


def find_mapping_csv():
    exact = [
        ROOT / "brahmi_mapping.csv",
        ROOT / "brahmi_mapping",
    ]

    path = find_first(exact)
    if path:
        return path

    # Screenshot showed the filename without .csv in Explorer.
    # Search recursively for any likely CSV mapping file.
    candidates = []

    for csv_path in ROOT.rglob("*.csv"):
        name = csv_path.name.lower()
        if (
            "brahmi" in name
            and (
                "mapping" in name
                or "unicode" in name
            )
        ):
            candidates.append(csv_path)

    if candidates:
        candidates.sort(
            key=lambda p: (
                "brahmi_mapping" not in p.name.lower(),
                len(p.parts),
                str(p).lower(),
            )
        )
        return candidates[0]

    return None


# ============================================================
# MODULE / IMAGE HELPERS
# ============================================================

def load_module(path, name):
    if not path.exists():
        raise FileNotFoundError(path)

    spec = importlib.util.spec_from_file_location(
        name,
        str(path)
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Cannot import {path}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_kannada_translation(records, mapping):
    """Build Kannada output directly from brahmi_mapping.csv.

    The CNN prediction is a class name.  The mapping CSV then provides
    the Kannada equivalent through its ``kannada_glyph`` column.

    No machine translation or external translation service is used here.
    """
    if not records:
        return ""

    output_lines = []
    missing = []

    line_numbers = sorted({int(r["line"]) for r in records})

    for line_no in line_numbers:
        line_records = [
            r for r in records
            if int(r["line"]) == line_no
        ]
        line_records.sort(
            key=lambda r: (
                int(r.get("x", 0)),
                int(r.get("char", 0)),
            )
        )

        kannada_chars = []
        for record in line_records:
            label = str(record.get("label", "")).strip()
            item = map_class_to_glyph(label, mapping)
            row = item.get("row", {}) or {}
            kannada = str(row.get("kannada_glyph", "")).strip()

            if kannada:
                kannada_chars.append(kannada)
            else:
                # Keep the output visibly aligned with the recognized class
                # if the CSV has no Kannada glyph for that class.
                kannada_chars.append(f"[{label}]")
                if label:
                    missing.append(label)

        output_lines.append("".join(kannada_chars))

    return "\n".join(output_lines), sorted(set(missing))


def load_gray(path):
    image = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        raise ValueError(
            f"Cannot read image: {path}"
        )

    return image


def load_binary(path):
    image = load_gray(path)

    _, binary = cv2.threshold(
        image,
        127,
        255,
        cv2.THRESH_BINARY
    )

    if np.mean(binary > 0) > 0.60:
        binary = cv2.bitwise_not(binary)

    return binary


def save_uploaded(uploaded):
    folder = Path(
        tempfile.mkdtemp(
            prefix="neolipi_"
        )
    )

    suffix = (
        Path(uploaded.name).suffix
        or ".png"
    )

    path = (
        folder /
        f"inscription{suffix}"
    )

    path.write_bytes(
        uploaded.getbuffer()
    )

    return path


# ============================================================
# SCRIPT CLASSIFIER
# ============================================================

def load_script_classes():
    candidates = [
        ROOT / "script_classes.json",
        ROOT / "class_indices.json",
        MODELS / "script_classes.json",
        MODELS / "class_indices.json",
    ]

    path = find_first(candidates)

    if path is None:
        return [
            "Brahmi",
            "Devanagari",
            "Tamil",
        ]

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:
        data = json.load(f)

    if isinstance(data, list):
        return [str(x) for x in data]

    if isinstance(data, dict):
        try:
            return [
                str(v)
                for _, v in sorted(
                    data.items(),
                    key=lambda item: int(item[0])
                )
            ]
        except (ValueError, TypeError):
            pass

        if all(
            isinstance(v, (int, float))
            for v in data.values()
        ):
            return [
                str(k)
                for k, _ in sorted(
                    data.items(),
                    key=lambda item: item[1]
                )
            ]

    raise ValueError(
        "Unsupported script class file."
    )


def classify_script(image_path):
    model_path = find_script_model()

    if model_path is None:
        raise FileNotFoundError(
            "Script classifier model was not found in models/."
        )

    import tensorflow as tf

    model = tf.keras.models.load_model(
        str(model_path)
    )

    input_shape = model.input_shape

    if isinstance(input_shape, list):
        input_shape = input_shape[0]

    if len(input_shape) != 4:
        raise ValueError(
            f"Unsupported script classifier input shape: {input_shape}"
        )

    target_h = int(
        input_shape[1]
        if input_shape[1] is not None
        else 224
    )

    target_w = int(
        input_shape[2]
        if input_shape[2] is not None
        else 224
    )

    channels = int(
        input_shape[3]
        if input_shape[3] is not None
        else 3
    )

    gray = load_gray(image_path)

    if channels == 3:
        image = cv2.cvtColor(
            gray,
            cv2.COLOR_GRAY2RGB
        )

        image = cv2.resize(
            image,
            (target_w, target_h),
            interpolation=cv2.INTER_AREA
        )

        x = (
            image.astype(np.float32)
            /
            255.0
        )

    elif channels == 1:
        image = cv2.resize(
            gray,
            (target_w, target_h),
            interpolation=cv2.INTER_AREA
        )

        x = (
            image.astype(np.float32)
            /
            255.0
        )

        x = x[..., np.newaxis]

    else:
        raise ValueError(
            f"Unsupported script classifier channels: {channels}"
        )

    x = x[np.newaxis, ...]

    probabilities = model.predict(
        x,
        verbose=0
    )[0]

    classes = load_script_classes()

    if len(probabilities) != len(classes):
        if len(probabilities) == 3:
            classes = [
                "Brahmi",
                "Devanagari",
                "Tamil",
            ]
        else:
            raise ValueError(
                "Script classifier output count does not "
                "match script classes."
            )

    order = np.argsort(
        probabilities
    )[::-1]

    top = [
        {
            "label": classes[int(i)],
            "confidence": float(
                probabilities[int(i)]
            )
        }
        for i in order[:3]
    ]

    best = int(order[0])

    return {
        "label": classes[best],
        "confidence": float(
            probabilities[best]
        ),
        "top": top,
        "input_shape": str(
            model.input_shape
        )
    }


# ============================================================
# BRAHMI CLASSES
# ============================================================

def load_brahmi_classes():
    path = find_brahmi_classes()

    if path is None:
        raise FileNotFoundError(
            "brahmi_classes.json was not found in models/."
        )

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:
        data = json.load(f)

    if isinstance(data, list):
        return [
            str(x)
            for x in data
        ]

    if isinstance(data, dict):
        try:
            return [
                str(v)
                for _, v in sorted(
                    data.items(),
                    key=lambda item: int(item[0])
                )
            ]
        except (ValueError, TypeError):
            pass

    raise ValueError(
        "Unsupported Brahmi class JSON."
    )


# ============================================================
# BRAHMI MAPPING CSV
# ============================================================

def clean_column(name):
    return (
        str(name)
        .strip()
        .lower()
        .replace("\ufeff", "")
        .replace(" ", "_")
        .replace("-", "_")
    )



def load_brahmi_mapping_csv():
    path = find_mapping_csv()

    if path is None:
        raise FileNotFoundError(
            "brahmi_mapping.csv "
            "was not found anywhere inside the project."
        )

    # Detect the delimiter. The file is CSV, but using the Sniffer
    # makes it tolerant to comma/semicolon/tab formatting.
    with open(
        path,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:
        sample = f.read(4096)

        f.seek(0)

        try:
            dialect = csv.Sniffer().sniff(
                sample,
                delimiters=",;\t|"
            )
        except csv.Error:
            dialect = csv.excel

        reader = csv.DictReader(
            f,
            dialect=dialect
        )

        raw_rows = list(reader)

    if not raw_rows:
        raise ValueError(
            f"Mapping CSV is empty: {path}"
        )

    # Normalize column names.
    rows = []

    for raw in raw_rows:
        normalized = {
            clean_column(k): (
                str(v).strip()
                if v is not None
                else ""
            )
            for k, v in raw.items()
            if k is not None
        }

        rows.append(normalized)

    # Exact uploaded CSV schema after normalization:
    # class_name, unicode_glyph, unicode_hex,
    # unicode_character_name, script_category.
    possible_label_columns = [
        "class_name",
        "class_label",
        "label",
        "class",
        "character",
        "name",
        "symbol",
    ]

    possible_glyph_columns = [
        "unicode_glyph",
        "brahmi_glyph",
        "glyph",
        "unicode_character",
        "unicode",
        "text",
    ]

    possible_hex_columns = [
        "unicode_hex",
        "codepoint",
        "code_point",
        "unicode_codepoint",
    ]


    label_column = None

    for col in possible_label_columns:
        if col in rows[0]:
            label_column = col
            break

    if label_column is None:
        raise ValueError(
            "Could not identify the class-label column in "
            f"{path.name}. Columns found: {list(rows[0].keys())}"
        )

    glyph_column = next(
        (
            c for c in possible_glyph_columns
            if c in rows[0]
        ),
        None
    )

    hex_column = next(
        (
            c for c in possible_hex_columns
            if c in rows[0]
        ),
        None
    )

    mapping = {}

    for row in rows:
        label = row.get(
            label_column,
            ""
        ).strip()

        if not label:
            continue

        glyph = (
            row.get(glyph_column, "").strip()
            if glyph_column
            else ""
        )

        codepoint = (
            row.get(hex_column, "").strip()
            if hex_column
            else ""
        )

        # If the CSV stores only a Unicode hex codepoint,
        # convert it to the actual Brahmi Unicode character(s)
        # when possible.
        if not glyph and codepoint:
            glyph = codepoint_to_text(
                codepoint
            )

        mapping[label] = {
            "class_label": label,
            "glyph": glyph,
            "unicode_hex": codepoint,
            "character_name": row.get(
                "unicode_character_name",
                ""
            ),
            "script_category": row.get(
                "script_category",
                ""
            ),
            "kannada_glyph": row.get(
                "kannada_glyph",
                ""
            ),
            "row": row,
        }

    if not mapping:
        raise ValueError(
            f"No usable class labels were found in {path}."
        )

    return path, mapping


def codepoint_to_text(value):
    value = str(value).strip()

    if not value:
        return ""

    # Supports:
    # U+11003
    # U+11024-U+11038
    # 11003
    # 11024-11038
    parts = value.replace(
        " ",
        ""
    ).upper().split("-")

    chars = []

    for part in parts:
        if part.startswith("U+"):
            part = part[2:]

        try:
            chars.append(
                chr(
                    int(
                        part,
                        16
                    )
                )
            )
        except ValueError:
            return ""

    return "".join(chars)


def normalize_label(label):
    return (
        str(label)
        .strip()
        .lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
    )


def map_class_to_glyph(
    label,
    mapping
):
    if label in mapping:
        return mapping[label]

    normalized = normalize_label(
        label
    )

    for key, value in mapping.items():
        if normalize_label(key) == normalized:
            return value

    return {
        "class_label": label,
        "glyph": "",
        "unicode_hex": "",
        "row": {},
    }


# ============================================================
# CHARACTER PREPROCESSING / OCR
# ============================================================

def prepare_character(
    image,
    size=75
):
    _, binary = cv2.threshold(
        image,
        0,
        255,
        cv2.THRESH_BINARY +
        cv2.THRESH_OTSU
    )

    ys, xs = np.where(
        binary > 0
    )

    if len(xs) == 0:
        return np.zeros(
            (size, size),
            dtype=np.uint8
        )

    x1 = int(
        xs.min()
    )

    x2 = int(
        xs.max()
    )

    y1 = int(
        ys.min()
    )

    y2 = int(
        ys.max()
    )

    binary = binary[
        y1:y2 + 1,
        x1:x2 + 1
    ]

    h, w = binary.shape

    pad = max(
        2,
        int(
            max(h, w) *
            0.15
        )
    )

    binary = cv2.copyMakeBorder(
        binary,
        pad,
        pad,
        pad,
        pad,
        cv2.BORDER_CONSTANT,
        value=0
    )

    h, w = binary.shape

    scale = (
        size *
        0.80
    ) / max(
        h,
        w
    )

    nw = max(
        1,
        int(
            round(
                w * scale
            )
        )
    )

    nh = max(
        1,
        int(
            round(
                h * scale
            )
        )
    )

    binary = cv2.resize(
        binary,
        (nw, nh),
        interpolation=cv2.INTER_AREA
    )

    canvas = np.zeros(
        (size, size),
        dtype=np.uint8
    )

    sx = (
        size -
        nw
    ) // 2

    sy = (
        size -
        nh
    ) // 2

    canvas[
        sy:sy + nh,
        sx:sx + nw
    ] = binary

    # Keep the existing model preprocessing behavior.
    density = (
        np.count_nonzero(
            canvas
        )
        /
        canvas.size
    )

    if density > 0.18:
        canvas = cv2.erode(
            canvas,
            np.ones(
                (2, 2),
                dtype=np.uint8
            ),
            iterations=1
        )

    return 255 - canvas


def recognize_character(
    crop,
    model,
    classes
):
    prepared = prepare_character(
        crop
    )

    input_shape = model.input_shape

    if isinstance(
        input_shape,
        list
    ):
        input_shape = input_shape[0]

    if len(input_shape) != 4:
        raise ValueError(
            f"Unsupported Brahmi model input shape: "
            f"{input_shape}"
        )

    target_h = int(
        input_shape[1]
        if input_shape[1] is not None
        else 75
    )

    target_w = int(
        input_shape[2]
        if input_shape[2] is not None
        else 75
    )

    channels = int(
        input_shape[3]
        if input_shape[3] is not None
        else 1
    )

    x = (
        prepared.astype(
            np.float32
        )
        /
        255.0
    )

    if x.shape != (
        target_h,
        target_w
    ):
        x = cv2.resize(
            x,
            (target_w, target_h),
            interpolation=cv2.INTER_AREA
        )

    if channels == 1:
        x = x[..., np.newaxis]

    elif channels == 3:
        x = np.repeat(
            x[..., np.newaxis],
            3,
            axis=-1
        )

    else:
        raise ValueError(
            f"Unsupported Brahmi model channels: {channels}"
        )

    x = x[np.newaxis, ...]

    probabilities = model.predict(
        x,
        verbose=0
    )[0]

    if len(probabilities) != len(classes):
        raise ValueError(
            "Brahmi model output count does not match "
            "brahmi_classes.json."
        )

    order = np.argsort(
        probabilities
    )[::-1]

    top5 = [
        {
            "label": classes[int(i)],
            "confidence": float(
                probabilities[int(i)]
            )
        }
        for i in order[:5]
    ]

    best = int(
        order[0]
    )

    return (
        prepared,
        classes[best],
        float(
            probabilities[best]
        ),
        top5
    )
# ============================================================
# DATASET CHARACTER SIMILARITY
# ============================================================

IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".bmp",
    ".tif", ".tiff", ".webp"
}


def _normalize_similarity_image(image, size=75):
    """Normalize a character image to a centered 75x75 binary shape."""
    if image is None or image.size == 0:
        return None

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Make a binary foreground mask.
    _, binary = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Force the character to white and background to black.
    # This handles both white-on-black and black-on-white inputs.
    if np.mean(binary > 0) > 0.50:
        binary = cv2.bitwise_not(binary)

    ys, xs = np.where(binary > 0)
    if len(xs) == 0:
        return np.zeros((size, size), dtype=np.uint8)

    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    character = binary[y1:y2, x1:x2]

    h, w = character.shape
    available = int(size * 0.80)
    scale = min(
        available / max(w, 1),
        available / max(h, 1)
    )

    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))

    character = cv2.resize(
        character,
        (nw, nh),
        interpolation=cv2.INTER_AREA
    )

    canvas = np.zeros((size, size), dtype=np.uint8)
    sx = (size - nw) // 2
    sy = (size - nh) // 2
    canvas[sy:sy + nh, sx:sx + nw] = character

    return canvas


@st.cache_resource(show_spinner=False)
def load_brahmi_dataset_gallery(dataset_path, max_images_per_class=40):
    """Load and normalize representative dataset images once per app run."""
    root = Path(dataset_path)

    if not root.exists():
        return {}

    gallery = {}

    for class_folder in sorted(root.iterdir(), key=lambda p: p.name.lower()):
        if not class_folder.is_dir():
            continue

        paths = [
            p for p in sorted(class_folder.iterdir(), key=lambda p: p.name.lower())
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ][:max_images_per_class]

        samples = []
        for path in paths:
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue

            normalized = _normalize_similarity_image(image)
            if normalized is not None:
                samples.append(normalized)

        if samples:
            gallery[class_folder.name] = samples

    return gallery


def _shape_similarity(a, b):
    """IoU/Jaccard similarity between two normalized binary shapes."""
    if a is None or b is None:
        return 0.0

    a_mask = a > 127
    b_mask = b > 127

    intersection = np.logical_and(a_mask, b_mask).sum()
    union = np.logical_or(a_mask, b_mask).sum()

    if union == 0:
        return 0.0

    return float(intersection / union)


def compare_with_dataset(crop, candidate_classes=None, max_images_per_class=40):
    """Compare an extracted character with class-wise dataset examples.

    The folder name is the dataset class. The class score is the mean of
    the best five image similarities in that class. This avoids letting a
    single unusual training image decide the result.
    """
    if not BRAHMI_DATASET.exists():
        return {
            "label": None,
            "similarity": 0.0,
            "top": [],
            "available": False,
        }

    gallery = load_brahmi_dataset_gallery(
        str(BRAHMI_DATASET),
        max_images_per_class=max_images_per_class,
    )

    if not gallery:
        return {
            "label": None,
            "similarity": 0.0,
            "top": [],
            "available": False,
        }

    query = _normalize_similarity_image(crop)
    if query is None:
        return {
            "label": None,
            "similarity": 0.0,
            "top": [],
            "available": True,
        }

    # By default compare against every class. This is important because
    # a wrong CNN prediction can otherwise hide the correct class.
    allowed = None
    if candidate_classes:
        allowed = {
            str(x).strip().lower()
            for x in candidate_classes
        }

    results = []

    for class_name, samples in gallery.items():
        if allowed is not None and class_name.lower() not in allowed:
            # We intentionally do NOT use this filter by default in the
            # pipeline below. It is kept for optional experiments.
            continue

        scores = [
            _shape_similarity(query, sample)
            for sample in samples
        ]

        if not scores:
            continue

        scores.sort(reverse=True)
        best_scores = scores[:min(5, len(scores))]

        results.append({
            "label": class_name,
            "similarity": float(np.mean(best_scores)),
            "best_single": float(scores[0]),
            "samples_used": len(samples),
        })

    results.sort(
        key=lambda item: item["similarity"],
        reverse=True,
    )

    if not results:
        return {
            "label": None,
            "similarity": 0.0,
            "top": [],
            "available": True,
        }

    return {
        "label": results[0]["label"],
        "similarity": results[0]["similarity"],
        "top": results[:5],
        "available": True,
    }


# ============================================================
# MANUAL CHARACTER SEGMENTATION HELPERS
# ============================================================

def _canvas_boxes_to_global(canvas_json, line_box, scale):
    """Convert drawable-canvas rectangles to global image boxes."""
    if not canvas_json or not canvas_json.get("objects"):
        return []

    lx1, ly1, _, _ = line_box
    boxes = []

    for obj in canvas_json["objects"]:
        if obj.get("type") != "rect":
            continue

        left = float(obj.get("left", 0.0))
        top = float(obj.get("top", 0.0))
        width = float(obj.get("width", 0.0))
        height = float(obj.get("height", 0.0))
        sx = float(obj.get("scaleX", 1.0))
        sy = float(obj.get("scaleY", 1.0))

        x = left / scale
        y = top / scale
        w = (width * sx) / scale
        h = (height * sy) / scale

        if w < 2 or h < 2:
            continue

        gx1 = int(round(lx1 + x))
        gy1 = int(round(ly1 + y))
        gx2 = int(round(lx1 + x + w))
        gy2 = int(round(ly1 + y + h))

        boxes.append((gx1, gy1, gx2, gy2))

    boxes.sort(key=lambda b: (b[0], b[1]))
    return boxes


def _records_from_manual_boxes(result, line_no, manual_boxes):
    """
    Re-run recognition for one manually corrected line.

    In addition to the existing recognition pipeline, this function
    saves every manually extracted binary character crop into:

        segmentation_brahmi_v6/manual_characters/

    File naming:
        line_01_char_001.png
        line_01_char_002.png
        line_02_char_001.png
        ...

    When the same line is manually corrected again, only that line's
    previous manually saved character images are removed and replaced.
    Other lines remain untouched.

    Existing binarization, segmentation, CNN recognition, dataset
    similarity, and final decision logic are preserved.
    """
    if not manual_boxes:
        return result["records"]

    model_path = find_brahmi_model()
    if model_path is None:
        raise FileNotFoundError("Brahmi OCR model was not found in models/.")

    class_list = load_brahmi_classes()
    _, mapping = load_brahmi_mapping_csv()

    import tensorflow as tf
    model = tf.keras.models.load_model(str(model_path))

    binary = result["binary"]
    H, W = binary.shape[:2]

    # ------------------------------------------------------------
    # Prepare manual-character output folder
    # ------------------------------------------------------------
    MANUAL_CHARACTERS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ------------------------------------------------------------
    # Remove previous manually saved crops for this line only.
    # This prevents duplicate files when the same line is corrected
    # more than once.
    # ------------------------------------------------------------
    old_line_files = sorted(
        MANUAL_CHARACTERS_DIR.glob(
            f"line_{int(line_no):02d}_char_*.png"
        )
    )

    for old_file in old_line_files:
        try:
            old_file.unlink()
        except OSError as exc:
            print(
                "[WARNING] Could not remove old manual character:",
                old_file,
                "|",
                exc
            )

    replacement = []

    # ------------------------------------------------------------
    # Process each manually drawn character box
    # ------------------------------------------------------------
    for char_no, (x1, y1, x2, y2) in enumerate(
        manual_boxes,
        start=1
    ):
        # Clip coordinates to image bounds.
        x1 = max(0, min(W - 1, int(x1)))
        y1 = max(0, min(H - 1, int(y1)))
        x2 = max(x1 + 1, min(W, int(x2)))
        y2 = max(y1 + 1, min(H, int(y2)))

        # Exact binary crop selected by the manual rectangle.
        crop = binary[y1:y2, x1:x2].copy()

        # --------------------------------------------------------
        # SAVE MANUALLY EXTRACTED CHARACTER
        # --------------------------------------------------------
        # Save the raw extracted binary crop BEFORE CNN preprocessing.
        # Therefore the image in this folder is the actual character
        # selected by the user during manual segmentation.
        manual_character_path = (
            MANUAL_CHARACTERS_DIR
            / (
                f"line_{int(line_no):02d}_"
                f"char_{int(char_no):03d}.png"
            )
        )

        if not cv2.imwrite(
            str(manual_character_path),
            crop
        ):
            raise IOError(
                "Could not save manually extracted character: "
                f"{manual_character_path}"
            )

        print(
            "[MANUAL CHARACTER SAVED]",
            manual_character_path
        )

        # --------------------------------------------------------
        # EXISTING CNN RECOGNITION
        # --------------------------------------------------------
        cnn_input, cnn_label, confidence, top5 = recognize_character(
            crop,
            model,
            class_list
        )

        # --------------------------------------------------------
        # EXISTING DATASET SIMILARITY
        # --------------------------------------------------------
        dataset_result = compare_with_dataset(
            crop,
            candidate_classes=None,
            max_images_per_class=40,
        )

        dataset_label = dataset_result["label"]
        dataset_similarity = dataset_result["similarity"]
        dataset_top = dataset_result.get("top", [])

        # --------------------------------------------------------
        # EXISTING DECISION ENGINE
        # --------------------------------------------------------
        final_label = cnn_label
        decision_source = "CNN"

        if dataset_label and dataset_top:
            best_score = float(
                dataset_top[0]["similarity"]
            )

            second_score = (
                float(dataset_top[1]["similarity"])
                if len(dataset_top) > 1
                else 0.0
            )

            margin = best_score - second_score

            if best_score >= 0.35 and margin >= 0.03:
                final_label = dataset_label
                decision_source = "DATASET_SIMILARITY"

        # --------------------------------------------------------
        # BRAHMI UNICODE MAPPING
        # --------------------------------------------------------
        mapping_item = map_class_to_glyph(
            final_label,
            mapping
        )

        # --------------------------------------------------------
        # Store result in the same format as the existing pipeline.
        # --------------------------------------------------------
        replacement.append({
            "global_index": None,
            "line": int(line_no),
            "char": int(char_no),
            "x": x1,
            "y": y1,
            "w": x2 - x1,
            "h": y2 - y1,
            "label": final_label,
            "cnn_label": cnn_label,
            "confidence": confidence,
            "recognition_status": recognition_status(confidence),
            "dataset_label": dataset_label,
            "dataset_similarity": dataset_similarity,
            "dataset_top": dataset_top,
            "decision_source": decision_source,
            "brahmi_glyph": mapping_item.get(
                "glyph",
                ""
            ),
            "unicode_hex": mapping_item.get(
                "unicode_hex",
                ""
            ),
            "unicode_character_name": mapping_item.get(
                "character_name",
                ""
            ),
            "raw": crop.copy(),
            "cnn": cnn_input,
            "top5": top5,
        })

    # ------------------------------------------------------------
    # Keep all other lines exactly as they were.
    # ------------------------------------------------------------
    other_records = [
        r
        for r in result["records"]
        if int(r["line"]) != int(line_no)
    ]

    merged = other_records + replacement

    # Keep inscription reading order: line top-to-bottom,
    # characters left-to-right.
    merged.sort(
        key=lambda r: (
            int(r["line"]),
            int(r.get("x", 0))
        )
    )

    # Rebuild global character numbering.
    for index, record in enumerate(
        merged,
        start=1
    ):
        record["global_index"] = index

    return merged


def render_manual_segmentation_editor(result):
    """Render an interactive per-line rectangle editor."""
    if not DRAWABLE_CANVAS_AVAILABLE:
        st.warning(
            "Manual segmentation editor needs streamlit-drawable-canvas. "
            "Run: pip install streamlit-drawable-canvas"
        )
        return

    st.subheader("4. Manual Segmentation Correction")
    st.write(
        "Use this when automatic segmentation merges, splits, misses, "
        "or cuts a character. Draw exactly one rectangle around each "
        "character that should be recognized."
    )

    line_boxes = result["line_boxes"]
    line_no = st.selectbox(
        "Select inscription line to correct",
        options=list(range(1, len(line_boxes) + 1)),
        format_func=lambda n: f"Line {n}",
        key="manual_line_selector",
    )

    mode = st.radio(
        "Correction mode",
        ["Adjust existing boxes", "Clear and redraw boxes"],
        horizontal=True,
        key="manual_segmentation_mode",
    )

    lx1, ly1, lx2, ly2 = line_boxes[line_no - 1]
    line = result["binary"][ly1:ly2, lx1:lx2]

    h, w = line.shape[:2]
    display_width = min(760, max(300, w))
    scale = display_width / float(max(w, 1))
    display_height = max(1, int(round(h * scale)))

    background = Image.fromarray(line).convert("RGB")
    background = background.resize(
        (display_width, display_height),
        Image.Resampling.NEAREST,
    )

    initial_objects = []
    if mode == "Adjust existing boxes":
        for record in result["records"]:
            if int(record["line"]) != line_no:
                continue

            x = (int(record["x"]) - lx1) * scale
            y = (int(record["y"]) - ly1) * scale
            bw = int(record["w"]) * scale
            bh = int(record["h"]) * scale

            initial_objects.append({
                "type": "rect",
                "left": x,
                "top": y,
                "width": max(2, bw),
                "height": max(2, bh),
                "fill": "rgba(0,255,0,0.05)",
                "stroke": "#00ff00",
                "strokeWidth": 2,
            })

    initial_drawing = {"version": "5.3.0", "objects": initial_objects}

    if mode == "Clear and redraw boxes":
        drawing_mode = "rect"
        initial_drawing = {"version": "5.3.0", "objects": []}
        help_text = "Draw one rectangle per character."
    else:
        drawing_mode = "transform"
        help_text = (
            "Click a green box, then drag it or resize it. "
            "Use Clear and redraw if you need to split a merged box."
        )

    st.caption(help_text)

    canvas_result = st_canvas(
        fill_color="rgba(0, 255, 0, 0.08)",
        stroke_width=2,
        stroke_color="#00ff00",
        background_image=background,
        update_streamlit=True,
        height=display_height,
        width=display_width,
        drawing_mode=drawing_mode,
        initial_drawing=initial_drawing,
        display_toolbar=True,
        key=f"manual_canvas_line_{line_no}_{mode}",
    )

    if canvas_result.json_data:
        object_count = sum(
            1 for obj in canvas_result.json_data.get("objects", [])
            if obj.get("type") == "rect"
        )
        st.caption(f"Boxes currently drawn: {object_count}")

    if st.button(
        "Apply Corrections & Re-run Recognition",
        type="primary",
        use_container_width=True,
        key="apply_manual_segmentation",
    ):
        if not canvas_result.json_data:
            st.error("Draw at least one character box first.")
            return

        boxes = _canvas_boxes_to_global(
            canvas_result.json_data,
            (lx1, ly1, lx2, ly2),
            scale,
        )

        if not boxes:
            st.error("No valid character rectangles were found.")
            return

        with st.spinner(
            f"Re-running recognition for Line {line_no}..."
        ):
            result["records"] = _records_from_manual_boxes(
                result,
                line_no,
                boxes,
            )

        st.session_state["result"] = result
        st.session_state["manual_correction_message"] = (
            f"Line {line_no} updated with {len(boxes)} manual character boxes."
        )
        st.rerun()

# ============================================================
# LINE BOX NORMALIZATION
# ============================================================

def normalize_line_box(box, image_shape):
    """
    Convert different line-box formats into:

        (x1, y1, x2, y2)

    Supported formats:

        (x1, y1, x2, y2)
        (y1, y2)
        ((x1, y1), (x2, y2))

    This keeps the rest of app_edit.py independent
    of the segmentation implementation.
    """

    H, W = image_shape[:2]

    # Convert numpy arrays / lists / tuples safely.
    if isinstance(box, np.ndarray):
        box = box.tolist()

    if not isinstance(box, (list, tuple)):
        raise ValueError(
            f"Invalid line box type: {type(box)}"
        )

    # --------------------------------------------------------
    # FORMAT 1:
    # (y1, y2)
    #
    # Full image width is used.
    # --------------------------------------------------------

    if len(box) == 2:

        a, b = box

        # Simple scalar pair
        if np.isscalar(a) and np.isscalar(b):

            y1 = int(a)
            y2 = int(b)

            y1 = max(0, min(H - 1, y1))
            y2 = max(y1 + 1, min(H, y2))

            return (
                0,
                y1,
                W,
                y2
            )

        # ----------------------------------------------------
        # FORMAT 2:
        # ((x1, y1), (x2, y2))
        # ----------------------------------------------------

        if (
            isinstance(a, (list, tuple))
            and
            isinstance(b, (list, tuple))
            and
            len(a) >= 2
            and
            len(b) >= 2
        ):

            x1 = int(a[0])
            y1 = int(a[1])
            x2 = int(b[0])
            y2 = int(b[1])

            x1 = max(0, min(W - 1, x1))
            y1 = max(0, min(H - 1, y1))
            x2 = max(x1 + 1, min(W, x2))
            y2 = max(y1 + 1, min(H, y2))

            return (
                x1,
                y1,
                x2,
                y2
            )

    # --------------------------------------------------------
    # FORMAT 3:
    # (x1, y1, x2, y2)
    # --------------------------------------------------------

    if len(box) >= 4:

        x1 = box[0]
        y1 = box[1]
        x2 = box[2]
        y2 = box[3]

        # Make sure none of these are nested tuples.
        if any(
            isinstance(v, (list, tuple, np.ndarray))
            for v in (x1, y1, x2, y2)
        ):

            # Handle:
            # ((x1,y1),(x2,y2),...)
            if (
                isinstance(x1, (list, tuple, np.ndarray))
                and
                isinstance(x2, (list, tuple, np.ndarray))
            ):
                p1 = x1
                p2 = x2

                x1 = p1[0]
                y1 = p1[1]

                x2 = p2[0]
                y2 = p2[1]

            else:
                raise ValueError(
                    f"Nested line box cannot be interpreted: {box}"
                )

        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

        # Clip to image.
        x1 = max(0, min(W - 1, x1))
        y1 = max(0, min(H - 1, y1))
        x2 = max(x1 + 1, min(W, x2))
        y2 = max(y1 + 1, min(H, y2))

        return (
            x1,
            y1,
            x2,
            y2
        )

    raise ValueError(
        f"Unsupported line-box format: {box}"
    )


def normalize_line_boxes(line_boxes, image_shape):
    """
    Normalize every detected line to:

        (x1, y1, x2, y2)
    """

    normalized = []

    for box in line_boxes:

        try:
            normalized_box = normalize_line_box(
                box,
                image_shape
            )

            normalized.append(
                normalized_box
            )

        except Exception as exc:

            print(
                "[WARNING] Skipping invalid line box:",
                box,
                "|",
                exc
            )

    normalized.sort(
        key=lambda b: (
            b[1],
            b[0]
        )
    )

    return normalized

# ============================================================
# FULL PIPELINE
# ============================================================


# ============================================================
# RECOGNITION CONFIDENCE STATUS
# ============================================================

def recognition_status(confidence):
    """
    Confidence is expected in the range 0.0 - 1.0.

    This does not alter the CNN prediction.
    It only labels how cautiously the prediction
    should be interpreted.
    """
    confidence = float(confidence)

    if confidence >= 0.85:
        return "HIGH"

    if confidence >= 0.65:
        return "MEDIUM"

    if confidence >= 0.45:
        return "LOW"

    return "UNCERTAIN"


def run_pipeline(uploaded):
    uploaded_path = save_uploaded(
        uploaded
    )

    # 1. SCRIPT CLASSIFICATION
    script_result = classify_script(
        uploaded_path
    )

    # 2. BINARIZATION
    bin_path = (
        ROOT /
        "binarization.py"
    )

    if not bin_path.exists():
        raise FileNotFoundError(
            "binarization.py was not found in the project folder."
        )

    bin_module = load_module(
        bin_path,
        "neolipi_binarization"
    )

    # Existing binarization.py writes outputs/08_FINAL_BINARY.png.
    bin_module.process(
        str(uploaded_path)
    )

    binary_path = (
        OUTPUTS /
        "08_FINAL_BINARY.png"
    )

    if not binary_path.exists():
        raise FileNotFoundError(
            "Binarization completed but "
            "outputs/08_FINAL_BINARY.png was not created."
        )

    binary = load_binary(
        binary_path
    )

    result = {
        "script": script_result,
        "binary": binary,
        "line_boxes": [],
        "records": [],
        "mapping_source": None,
    }

    # Detailed recognition currently for Brahmi only.
    if (
        script_result["label"].lower()
        !=
        "brahmi"
    ):
        result["message"] = (
            f"Detected {script_result['label']}. "
            "For the current project sprint, the detailed "
            "character recognition/mapping stage is enabled "
            "for Brahmi."
        )
        return result

    # 3. CHARACTER SEGMENTATION
    # Prefer the original segmentation module because this is
    # the version you said gives the better segmentation.
    # Fall back to V6 if the original file is absent.
    segmentation_path = ROOT / "character_segmentation_1dup.py"

    if not segmentation_path.exists():
        raise FileNotFoundError(
            "character_segmentation_1dup.py was not found "
            "in the project folder."
        )

    segmentation = load_module(
        segmentation_path,
        "neolipi_segmentation"
    )

    # character_segmentation_1.py V6 returns line boxes as:
    #     (y1, y2)
    # The app pipeline needs global boxes as:
    #     (x1, y1, x2, y2)
    # Since V6 detects lines across the full image width,
    # convert each (y1, y2) into a full-width global box.
    detected = segmentation.detect_lines(binary)

    # Support all segmentation versions:
    #   [(y1, y2), ...]
    #   [(x1, y1, x2, y2), ...]
    #   (line_boxes, peaks, signal)
    if isinstance(detected, tuple) and len(detected) == 3:
        detected_lines = detected[0]
    else:
        detected_lines = detected

    result["line_boxes"] = normalize_line_boxes(
        detected_lines,
        binary.shape
    )

    if not result["line_boxes"]:
        raise ValueError(
            "No inscription lines were detected."
        )

    # 4. BRAHMI OCR RESOURCES
    model_path = find_brahmi_model()

    if model_path is None:
        raise FileNotFoundError(
            "Brahmi OCR model was not found in models/."
        )

    class_list = load_brahmi_classes()

    mapping_path, mapping = (
        load_brahmi_mapping_csv()
    )

    result["mapping_source"] = str(
        mapping_path
    )

    import tensorflow as tf

    model = tf.keras.models.load_model(
        str(model_path)
    )

    # 5. EXTRACT -> RECOGNIZE -> MAP TO BRAHMI UNICODE
    global_index = 1

    # Use the normalized boxes stored in result.
    line_boxes = result["line_boxes"]

    for line_no, line_box in enumerate(
        line_boxes,
        start=1
    ):
        lx1, ly1, lx2, ly2 = line_box

        line = binary[
            ly1:ly2,
            lx1:lx2
        ].copy()

        segmented = segmentation.segment_line(line)

        # Support either:
        #   components
        # or:
        #   (line_roi, components)
        if (
            isinstance(segmented, tuple)
            and len(segmented) == 2
        ):
            components = segmented[1]
        else:
            components = segmented

        if components is None:
            components = []

        for char_no, component in enumerate(
            components,
            start=1
        ):
            crop = (
                segmentation.extract_character(
                    binary,
                    line_box,
                    component
                )
            )

            (
                cnn_input,
                cnn_label,
                confidence,
                top5
            ) = recognize_character(
                crop,
                model,
                class_list
            )

            # ----------------------------------------------------
            # DATASET-BASED CLASS VERIFICATION
            # ----------------------------------------------------
            # Compare the extracted character against the actual
            # class folders in RecognizerDataset_150_210.
            # We compare against ALL classes rather than only the
            # CNN Top-5, because the CNN itself may be wrong.
            dataset_result = compare_with_dataset(
                crop,
                candidate_classes=None,
                max_images_per_class=40,
            )

            dataset_label = dataset_result["label"]
            dataset_similarity = dataset_result["similarity"]

            # Dataset similarity becomes the final class only when
            # it has a reasonably strong match AND beats the second
            # best dataset class by a useful margin. Otherwise the
            # existing CNN result is retained.
            final_label = cnn_label
            decision_source = "CNN"

            dataset_top = dataset_result.get("top", [])

            if dataset_label and dataset_top:
                best_score = float(dataset_top[0]["similarity"])
                second_score = (
                    float(dataset_top[1]["similarity"])
                    if len(dataset_top) > 1
                    else 0.0
                )
                margin = best_score - second_score

                if best_score >= 0.35 and margin >= 0.03:
                    final_label = dataset_label
                    decision_source = "DATASET_SIMILARITY"

            mapping_item = map_class_to_glyph(
                final_label,
                mapping
            )

            recognition_level = recognition_status(
                confidence
            )

            x, y, w, h, _ = component

            result["records"].append({
                "global_index": global_index,
                "line": line_no,
                "char": char_no,
                "x": int(lx1 + x),
                "y": int(ly1 + y),
                "w": int(w),
                "h": int(h),
                # Final class used for Unicode mapping.
                "label": final_label,

                # Keep the raw CNN result for debugging.
                "cnn_label": cnn_label,
                "confidence": confidence,
                "recognition_status": recognition_level,

                # Dataset similarity evidence.
                "dataset_label": dataset_label,
                "dataset_similarity": dataset_similarity,
                "dataset_top": dataset_top,
                "decision_source": decision_source,

                "brahmi_glyph": mapping_item.get(
                    "glyph",
                    ""
                ),
                "unicode_hex": mapping_item.get(
                    "unicode_hex",
                    ""
                ),
                "unicode_character_name": mapping_item.get(
                    "character_name",
                    ""
                ),
                "raw": crop,
                "cnn": cnn_input,
                "top5": top5,
            })

            global_index += 1

    return result

# ============================================================
# BRAHMI LABEL -> READABLE TEXT
# ============================================================

def build_readable_text(records):

    if not records:
        return ""

    readable_lines = []

    line_numbers = sorted(
        set(
            int(r["line"])
            for r in records
        )
    )

    for line_no in line_numbers:

        line_records = [
            r
            for r in records
            if int(r["line"]) == line_no
        ]

        # IMPORTANT:
        # Sort characters from LEFT -> RIGHT
        line_records.sort(
            key=lambda r: (
                int(r.get("x", 0)),
                int(r.get("char", 0))
            )
        )

        labels = []

        for r in line_records:

            label = str(
                r.get("label", "")
            ).strip()

            if label:
                labels.append(label)

        if labels:
            readable_lines.append(
                " ".join(labels)
            )

    return "\n".join(
        readable_lines
    )



# ============================================================
# NEW ALL-CLASS CNN MAPPING PIPELINE
# ============================================================
# This block intentionally overrides only the recognition/storage
# functions. The existing binarization and segmentation modules are
# preserved unchanged.

ALL_CLASS_CNN_MODEL_PATH = Path(
    r"D:\NeoLipi_VSCODE\NeoLipi_CRNN\cnn_all_classes_stone_augmented_best.keras"
)

ALL_CLASS_NAMES_PATH = Path(
    r"D:\NeoLipi_VSCODE\NeoLipi_CRNN\all_class_names.txt"
)

MODEL_IMG_SIZE = 64

MANUAL_CHARACTER_ROOT = (
    ROOT / "segmentation_brahmi_v6" / "manual_characters"
)
MANUAL_CHARACTER_ROOT.mkdir(parents=True, exist_ok=True)


def find_brahmi_model():
    """Return the exact all-class CNN used by the working test script."""
    if not ALL_CLASS_CNN_MODEL_PATH.exists():
        raise FileNotFoundError(
            "All-class CNN model was not found:\n"
            f"{ALL_CLASS_CNN_MODEL_PATH}"
        )
    return ALL_CLASS_CNN_MODEL_PATH


def load_brahmi_classes():
    """Load class order exactly as saved during all-class training."""
    if not ALL_CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(
            "all_class_names.txt was not found:\n"
            f"{ALL_CLASS_NAMES_PATH}"
        )

    with open(ALL_CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        classes = [line.strip() for line in f if line.strip()]

    if not classes:
        raise ValueError("all_class_names.txt is empty.")

    return classes


@st.cache_resource(show_spinner=False)
def load_all_class_cnn():
    """Load the working all-class CNN once per Streamlit process."""
    import tensorflow as tf

    model_path = find_brahmi_model()
    model = tf.keras.models.load_model(str(model_path))
    classes = load_brahmi_classes()

    output_shape = model.output_shape
    if isinstance(output_shape, list):
        output_shape = output_shape[0]

    output_count = int(output_shape[-1])
    if output_count != len(classes):
        raise ValueError(
            "CNN output count does not match all_class_names.txt: "
            f"model={output_count}, classes={len(classes)}"
        )

    input_shape = model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]

    if len(input_shape) != 4:
        raise ValueError(
            f"Unsupported CNN input shape: {input_shape}"
        )

    return model, classes


def create_character_run_dir():
    """Create a separate extraction/preprocessing folder per inscription run."""
    import datetime

    stamp = datetime.datetime.now().strftime("stone_%Y%m%d_%H%M%S_%f")[:-3]
    run_dir = MANUAL_CHARACTER_ROOT / stamp
    extracted_dir = run_dir / "extracted"
    preprocessed_dir = run_dir / "preprocessed"
    extracted_dir.mkdir(parents=True, exist_ok=True)
    preprocessed_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, extracted_dir, preprocessed_dir


def save_extracted_character(run_dir, line_no, char_no, crop):
    """Save the exact binary crop produced by the existing segmentation."""
    extracted_dir = Path(run_dir) / "extracted"
    extracted_dir.mkdir(parents=True, exist_ok=True)

    path = extracted_dir / (
        f"line_{int(line_no):02d}_char_{int(char_no):03d}.png"
    )

    if not cv2.imwrite(str(path), crop):
        raise IOError(f"Could not save extracted character: {path}")

    return path


def preprocess_character_for_all_class_cnn(image, size=MODEL_IMG_SIZE):
    """Match the preprocessing used to prepare the augmented stone characters.

    Output:
        64x64, grayscale, black character on white background.
    """
    if image is None or image.size == 0:
        raise ValueError("Empty character image.")

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Otsu threshold.
    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    # Make the foreground the white character temporarily.
    # If white occupies most of the image, the character is assumed
    # to be black on white and is inverted.
    white_pixels = cv2.countNonZero(binary)
    total_pixels = binary.shape[0] * binary.shape[1]
    black_pixels = total_pixels - white_pixels

    if white_pixels > black_pixels:
        binary = cv2.bitwise_not(binary)

    # Find character foreground.
    coords = cv2.findNonZero(binary)
    if coords is None:
        return np.full((size, size), 255, dtype=np.uint8)

    x, y, w, h = cv2.boundingRect(coords)
    cropped = binary[y:y + h, x:x + w]

    # Padding used in the standalone preprocessing pipeline.
    padding = 8
    cropped = cv2.copyMakeBorder(
        cropped,
        padding,
        padding,
        padding,
        padding,
        cv2.BORDER_CONSTANT,
        value=0,
    )

    h, w = cropped.shape
    square_size = max(h, w)
    square = np.zeros(
        (square_size, square_size),
        dtype=np.uint8,
    )

    y_offset = (square_size - h) // 2
    x_offset = (square_size - w) // 2

    square[
        y_offset:y_offset + h,
        x_offset:x_offset + w,
    ] = cropped

    resized = cv2.resize(
        square,
        (size, size),
        interpolation=cv2.INTER_AREA,
    )

    # Final format expected by the working model:
    # white background + black character.
    final_image = cv2.bitwise_not(resized)

    _, final_image = cv2.threshold(
        final_image,
        127,
        255,
        cv2.THRESH_BINARY,
    )

    return final_image


def preprocess_extracted_folder(run_dir):
    """Preprocess every extracted image and save it to preprocessed/."""
    run_dir = Path(run_dir)
    extracted_dir = run_dir / "extracted"
    preprocessed_dir = run_dir / "preprocessed"
    preprocessed_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(
        [
            p for p in extracted_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ],
        key=lambda p: p.name.lower(),
    )

    processed = {}

    for image_path in image_files:
        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_GRAYSCALE,
        )
        if image is None:
            continue

        prepared = preprocess_character_for_all_class_cnn(image)
        output_path = preprocessed_dir / (
            image_path.stem + "_preprocessed.png"
        )

        if not cv2.imwrite(str(output_path), prepared):
            raise IOError(
                f"Could not save preprocessed character: {output_path}"
            )

        processed[image_path.name] = output_path

    return processed


def predict_preprocessed_folder(run_dir):
    """Run the all-class CNN on the saved preprocessed folder."""
    model, classes = load_all_class_cnn()
    preprocessed_dir = Path(run_dir) / "preprocessed"

    predictions = {}

    image_files = sorted(
        [
            p for p in preprocessed_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ],
        key=lambda p: p.name.lower(),
    )

    if not image_files:
        return predictions

    batch_images = []
    valid_paths = []

    for image_path in image_files:
        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_GRAYSCALE,
        )
        if image is None:
            continue

        image = cv2.resize(
            image,
            (MODEL_IMG_SIZE, MODEL_IMG_SIZE),
            interpolation=cv2.INTER_AREA,
        )

        x = image.astype(np.float32) / 255.0
        x = x[..., np.newaxis]

        batch_images.append(x)
        valid_paths.append(image_path)

    if not batch_images:
        return predictions

    batch = np.stack(batch_images, axis=0)
    probabilities = model.predict(batch, verbose=0)

    for image_path, probs in zip(valid_paths, probabilities):
        order = np.argsort(probs)[::-1]
        best = int(order[0])

        top5 = [
            {
                "label": classes[int(i)],
                "confidence": float(probs[int(i)]),
            }
            for i in order[:5]
        ]

        predictions[image_path.stem.replace("_preprocessed", "")] = {
            "label": classes[best],
            "confidence": float(probs[best]),
            "top5": top5,
            "preprocessed_path": str(image_path),
        }

    return predictions


def recognize_character(crop, model=None, classes=None):
    """Compatibility wrapper: preprocess one crop and classify it."""
    prepared = preprocess_character_for_all_class_cnn(crop)

    if model is None or classes is None:
        model, classes = load_all_class_cnn()

    x = prepared.astype(np.float32) / 255.0
    x = x[..., np.newaxis]
    x = x[np.newaxis, ...]

    probabilities = model.predict(x, verbose=0)[0]
    order = np.argsort(probabilities)[::-1]

    top5 = [
        {
            "label": classes[int(i)],
            "confidence": float(probabilities[int(i)]),
        }
        for i in order[:5]
    ]

    best = int(order[0])

    return (
        prepared,
        classes[best],
        float(probabilities[best]),
        top5,
    )


def _prediction_key(line_no, char_no):
    return f"line_{int(line_no):02d}_char_{int(char_no):03d}"


def _records_from_manual_boxes(result, line_no, manual_boxes):
    """Replace one line's extracted crops, preprocess the whole run folder,
    and map the resulting preprocessed images with the all-class CNN."""
    if not manual_boxes:
        return result["records"]

    run_dir = result.get("run_dir")
    if not run_dir:
        run_dir, _, _ = create_character_run_dir()
        result["run_dir"] = str(run_dir)

    run_dir = Path(run_dir)
    extracted_dir = run_dir / "extracted"
    preprocessed_dir = run_dir / "preprocessed"
    extracted_dir.mkdir(parents=True, exist_ok=True)
    preprocessed_dir.mkdir(parents=True, exist_ok=True)

    binary = result["binary"]
    H, W = binary.shape[:2]

    # Remove old extracted/preprocessed images belonging to this line.
    for pattern in (
        f"line_{int(line_no):02d}_char_*.png",
        f"line_{int(line_no):02d}_char_*_preprocessed.png",
    ):
        for old_file in list(extracted_dir.glob(pattern)) + list(preprocessed_dir.glob(pattern)):
            try:
                old_file.unlink()
            except OSError:
                pass

    replacement = []

    for char_no, (x1, y1, x2, y2) in enumerate(manual_boxes, start=1):
        x1 = max(0, min(W - 1, int(x1)))
        y1 = max(0, min(H - 1, int(y1)))
        x2 = max(x1 + 1, min(W, int(x2)))
        y2 = max(y1 + 1, min(H, int(y2)))

        crop = binary[y1:y2, x1:x2].copy()
        save_extracted_character(
            run_dir,
            line_no,
            char_no,
            crop,
        )

    # The requirement is that mapping always consumes the complete
    # preprocessed folder, not an in-memory crop.
    preprocess_extracted_folder(run_dir)
    predictions = predict_preprocessed_folder(run_dir)

    _, mapping = load_brahmi_mapping_csv()

    for char_no, (x1, y1, x2, y2) in enumerate(manual_boxes, start=1):
        key = _prediction_key(line_no, char_no)
        prediction = predictions.get(key)

        if prediction is None:
            continue

        final_label = prediction["label"]
        mapping_item = map_class_to_glyph(final_label, mapping)

        replacement.append({
            "global_index": None,
            "line": int(line_no),
            "char": int(char_no),
            "x": x1,
            "y": y1,
            "w": x2 - x1,
            "h": y2 - y1,
            "label": final_label,
            "cnn_label": final_label,
            "confidence": prediction["confidence"],
            "recognition_status": recognition_status(
                prediction["confidence"]
            ),
            "dataset_label": None,
            "dataset_similarity": 0.0,
            "dataset_top": [],
            "decision_source": "ALL_CLASS_CNN",
            "brahmi_glyph": mapping_item.get("glyph", ""),
            "unicode_hex": mapping_item.get("unicode_hex", ""),
            "unicode_character_name": mapping_item.get(
                "character_name", ""
            ),
            "raw": crop,
            "cnn": cv2.imread(
                str(preprocessed_dir / f"{key}_preprocessed.png"),
                cv2.IMREAD_GRAYSCALE,
            ),
            "top5": prediction["top5"],
            "extracted_path": str(
                extracted_dir / f"{key}.png"
            ),
            "preprocessed_path": prediction["preprocessed_path"],
        })

    other_records = [
        r for r in result["records"]
        if int(r["line"]) != int(line_no)
    ]

    merged = other_records + replacement
    merged.sort(
        key=lambda r: (
            int(r["line"]),
            int(r.get("x", 0)),
        )
    )

    for index, record in enumerate(merged, start=1):
        record["global_index"] = index

    return merged


def run_pipeline(uploaded):
    """NeoLipi pipeline using the working all-class CNN mapping model."""
    uploaded_path = save_uploaded(uploaded)

    # 1. SCRIPT CLASSIFICATION - unchanged.
    script_result = classify_script(uploaded_path)

    # 2. BINARIZATION - unchanged.
    bin_path = ROOT / "binarization.py"
    if not bin_path.exists():
        raise FileNotFoundError(
            "binarization.py was not found in the project folder."
        )

    bin_module = load_module(
        bin_path,
        "neolipi_binarization_new",
    )
    bin_module.process(str(uploaded_path))

    binary_path = OUTPUTS / "08_FINAL_BINARY.png"
    if not binary_path.exists():
        raise FileNotFoundError(
            "Binarization completed but outputs/08_FINAL_BINARY.png "
            "was not created."
        )

    binary = load_binary(binary_path)

    result = {
        "script": script_result,
        "binary": binary,
        "line_boxes": [],
        "records": [],
        "mapping_source": None,
        "run_dir": None,
        "extracted_dir": None,
        "preprocessed_dir": None,
    }

    if script_result["label"].lower() != "brahmi":
        result["message"] = (
            f"Detected {script_result['label']}. "
            "For the current project sprint, detailed character "
            "recognition/mapping is enabled for Brahmi."
        )
        return result

    # 3. SEGMENTATION - existing working segmentation module unchanged.
    segmentation_path = ROOT / "character_segmentation_1dup.py"
    if not segmentation_path.exists():
        raise FileNotFoundError(
            "character_segmentation_1dup.py was not found in the project folder."
        )

    segmentation = load_module(
        segmentation_path,
        "neolipi_segmentation_new",
    )

    detected = segmentation.detect_lines(binary)

    if isinstance(detected, tuple) and len(detected) == 3:
        detected_lines = detected[0]
    else:
        detected_lines = detected

    result["line_boxes"] = normalize_line_boxes(
        detected_lines,
        binary.shape,
    )

    if not result["line_boxes"]:
        raise ValueError("No inscription lines were detected.")

    # 4. Exact working all-class CNN + class file.
    model, class_list = load_all_class_cnn()
    mapping_path, mapping = load_brahmi_mapping_csv()
    result["mapping_source"] = str(mapping_path)

    # 5. Create this inscription's extraction/preprocessing folders.
    run_dir, extracted_dir, preprocessed_dir = create_character_run_dir()
    result["run_dir"] = str(run_dir)
    result["extracted_dir"] = str(extracted_dir)
    result["preprocessed_dir"] = str(preprocessed_dir)

    # ------------------------------------------------------------
    # STEP A: EXISTING SEGMENTATION -> SAVE ALL RAW CHARACTERS
    # ------------------------------------------------------------
    components_by_line = {}

    for line_no, line_box in enumerate(
        result["line_boxes"],
        start=1,
    ):
        lx1, ly1, lx2, ly2 = line_box
        line = binary[ly1:ly2, lx1:lx2].copy()

        segmented = segmentation.segment_line(line)
        if isinstance(segmented, tuple) and len(segmented) == 2:
            components = segmented[1]
        else:
            components = segmented

        if components is None:
            components = []

        components_by_line[line_no] = []

        for char_no, component in enumerate(components, start=1):
            crop = segmentation.extract_character(
                binary,
                line_box,
                component,
            )

            extracted_path = save_extracted_character(
                run_dir,
                line_no,
                char_no,
                crop,
            )

            x, y, w, h, _ = component
            components_by_line[line_no].append({
                "char": char_no,
                "component": component,
                "crop": crop,
                "extracted_path": extracted_path,
                "x": int(lx1 + x),
                "y": int(ly1 + y),
                "w": int(w),
                "h": int(h),
            })

    # ------------------------------------------------------------
    # STEP B: PREPROCESS THE WHOLE EXTRACTED FOLDER
    # ------------------------------------------------------------
    preprocess_extracted_folder(run_dir)

    # ------------------------------------------------------------
    # STEP C: MAPPING MUST USE THE PREPROCESSED FOLDER
    # ------------------------------------------------------------
    predictions = predict_preprocessed_folder(run_dir)

    global_index = 1

    for line_no in sorted(components_by_line):
        for item in components_by_line[line_no]:
            char_no = item["char"]
            key = _prediction_key(line_no, char_no)
            prediction = predictions.get(key)

            if prediction is None:
                continue

            final_label = prediction["label"]
            mapping_item = map_class_to_glyph(
                final_label,
                mapping,
            )

            preprocessed_path = Path(
                prediction["preprocessed_path"]
            )
            cnn_input = cv2.imread(
                str(preprocessed_path),
                cv2.IMREAD_GRAYSCALE,
            )

            result["records"].append({
                "global_index": global_index,
                "line": line_no,
                "char": char_no,
                "x": item["x"],
                "y": item["y"],
                "w": item["w"],
                "h": item["h"],
                "label": final_label,
                "cnn_label": final_label,
                "confidence": prediction["confidence"],
                "recognition_status": recognition_status(
                    prediction["confidence"]
                ),
                "dataset_label": None,
                "dataset_similarity": 0.0,
                "dataset_top": [],
                "decision_source": "ALL_CLASS_CNN",
                "brahmi_glyph": mapping_item.get("glyph", ""),
                "unicode_hex": mapping_item.get("unicode_hex", ""),
                "unicode_character_name": mapping_item.get(
                    "character_name", ""
                ),
                "raw": item["crop"].copy(),
                "cnn": cnn_input,
                "top5": prediction["top5"],
                "extracted_path": str(item["extracted_path"]),
                "preprocessed_path": str(preprocessed_path),
            })

            global_index += 1

    return result


# ============================================================
# STREAMLIT UI
# ============================================================

st.title(
    "NeoLipi — Ancient Inscription Recognition"
)

st.write(
    "Upload a stone inscription and click Analyze."
)

uploaded = st.file_uploader(
    "Upload Stone Inscription Image",
    type=[
        "png",
        "jpg",
        "jpeg",
        "bmp",
        "webp",
    ],
)

if uploaded is None:
    st.info(
        "Please upload an inscription image."
    )
    st.stop()


st.subheader(
    "Input Image"
)

st.image(
    uploaded,
    caption=uploaded.name,
    width='stretch',
)


if st.button(
    "Analyze",
    type="primary",
    width='stretch',
):
    try:
        with st.spinner(
            "Running NeoLipi..."
        ):
            st.session_state[
                "result"
            ] = run_pipeline(
                uploaded
            )

    except Exception as exc:
        st.error(
            "Analysis failed."
        )
        st.exception(exc)
        st.stop()


if "result" not in st.session_state:
    st.stop()


result = st.session_state[
    "result"
]

# The current run owns its extracted/preprocessed folders.
# Point the manual editor at that run so corrections replace only
# the selected line inside the same inscription run.
if result.get("run_dir"):
    MANUAL_CHARACTERS_DIR = Path(result["run_dir"]) / "extracted"
    MANUAL_CHARACTERS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. SCRIPT CLASSIFICATION
# ============================================================

st.subheader(
    "1. Script Classification"
)

c1, c2 = st.columns(2)

with c1:
    st.success(
        f"Detected Script: "
        f"{result['script']['label']}"
    )

with c2:
    st.metric(
        "Confidence",
        f"{result['script']['confidence'] * 100:.2f}%"
    )

st.dataframe(
    pd.DataFrame([
        {
            "Script": item["label"],
            "Confidence": (
                f"{item['confidence'] * 100:.2f}%"
            ),
        }
        for item in result["script"]["top"]
    ]),
    hide_index=True,
    width='stretch',
)

st.caption(
    f"Script classifier input: "
    f"{result['script']['input_shape']}"
)


# ============================================================
# 2. BINARY
# ============================================================

st.subheader(
    "2. Binary Image"
)

st.image(
    result["binary"],
    caption="Generated Binary Mask",
    width='stretch',
)


if (
    result["script"]["label"].lower()
    !=
    "brahmi"
):
    st.warning(
        result["message"]
    )
    st.stop()


# ============================================================
# 3. SEGMENTATION
# ============================================================

st.subheader(
    "3. Character Segmentation"
)

st.write(
    f"Detected lines: "
    f"**{len(result['line_boxes'])}**"
)

line_image = cv2.cvtColor(
    result["binary"],
    cv2.COLOR_GRAY2RGB
)

for i, (
    x1,
    y1,
    x2,
    y2
) in enumerate(
    result["line_boxes"],
    start=1
):
    cv2.rectangle(
        line_image,
        (x1, y1),
        (x2 - 1, y2 - 1),
        (255, 255, 0),
        2,
    )

    cv2.putText(
        line_image,
        f"Line {i}",
        (x1, max(15, y1 - 3)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 0),
        1,
        cv2.LINE_AA,
    )

st.image(
    line_image,
    caption="Detected Inscription Lines",
    width='stretch',
)


# ============================================================
# 4. MANUAL SEGMENTATION CORRECTION
# ============================================================

render_manual_segmentation_editor(result)

# Show where manually extracted character images are being saved.
manual_character_files = sorted(
    MANUAL_CHARACTERS_DIR.glob("*.png")
)

if manual_character_files:
    st.success(
        f"{len(manual_character_files)} extracted character image(s) "
        "saved for this inscription run."
    )


if result.get("extracted_dir"):
    st.caption(f"Extracted characters: {result['extracted_dir']}")
if result.get("preprocessed_dir"):
    st.caption(f"Preprocessed characters used for mapping: {result['preprocessed_dir']}")

# ============================================================
# 5. CHARACTER EXTRACTION
# ============================================================

st.subheader(
    "4. Character Extraction"
)

box_image = cv2.cvtColor(
    result["binary"],
    cv2.COLOR_GRAY2RGB
)

for record in result["records"]:

    x1 = int(record["x"])
    y1 = int(record["y"])

    x2 = (
        x1 +
        int(record["w"])
    )

    y2 = (
        y1 +
        int(record["h"])
    )

    number = int(
        record["global_index"]
    )

    # Character bounding box
    cv2.rectangle(
        box_image,
        (x1, y1),
        (max(x1 + 1, x2 - 1),
         max(y1 + 1, y2 - 1)),
        (0, 255, 0),
        1,
    )

    # Character number
    cv2.putText(
        box_image,
        str(number),
        (
            x1,
            max(12, y1 - 2)
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.36,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )

st.image(
    box_image,
    caption="Character Bounding Boxes",
    width="stretch",
)

# ============================================================
# 5. CHARACTER -> BRAHMI UNICODE MAPPING
# ============================================================

# ============================================================
# CHARACTER QUALITY DIAGNOSTICS
# ============================================================

def character_quality(record):
    """
    Analyze the extracted character crop.

    This does NOT modify the crop.
    It is only a diagnostic tool to identify
    suspicious segmentation results.
    """

    crop = record["raw"]

    if crop is None or crop.size == 0:
        return {
            "status": "INVALID",
            "aspect_ratio": 0.0,
            "foreground_ratio": 0.0,
            "width": 0,
            "height": 0,
        }

    h, w = crop.shape[:2]

    foreground = (
        np.count_nonzero(crop > 127)
    )

    foreground_ratio = (
        foreground / float(crop.size)
    )

    aspect_ratio = (
        w / float(h)
        if h > 0
        else 0.0
    )

    # Conservative diagnostic rules.
    #
    # These DO NOT remove characters.
    # They only mark suspicious crops.

    status = "GOOD"

    if w < 3 or h < 5:
        status = "TOO SMALL"

    elif foreground_ratio < 0.015:
        status = "MOSTLY NOISE"

    elif foreground_ratio > 0.70:
        status = "TOO DENSE"

    elif aspect_ratio > 4.5:
        status = "VERY WIDE"

    elif aspect_ratio < 0.15:
        status = "VERY NARROW"

    return {
        "status": status,
        "aspect_ratio": aspect_ratio,
        "foreground_ratio": foreground_ratio,
        "width": w,
        "height": h,
    }




# ============================================================
# 6. CHARACTER RECOGNITION + BRAHMI UNICODE
# ============================================================

st.subheader(
    "6. Character Recognition + Brahmi Unicode Mapping"
)

table_rows = []

for r in result["records"]:

    quality = character_quality(r)

    table_rows.append({

        "#":
            r["global_index"],

        "Line":
            r["line"],

        "Char":
            r["char"],

        "Predicted Class":
            r["label"],

        "Brahmi Unicode":
            r["brahmi_glyph"]
            or "N/A",

        "Unicode Name":
            r.get(
                "unicode_character_name",
                ""
            ),

        "Confidence":
            f"{r['confidence'] * 100:.2f}%",

        "Recognition":
            r.get(
                "recognition_status",
                recognition_status(
                    r["confidence"]
                )
            ),

        "Segmentation":
            quality["status"],

        "W":
            r["w"],

        "H":
            r["h"],
    })

table = pd.DataFrame(
    table_rows
)

st.dataframe(
    table,
    hide_index=True,
    width="stretch",
)

st.caption(
    "The Unicode value comes directly from "
    "brahmi_mapping.csv."
)


# ============================================================
# 7. CHARACTER INSPECTOR
# ============================================================

if result["records"]:

    st.subheader(
        "6. Character Inspector"
    )

    st.write(
        "Inspect each extracted character before judging "
        "the CNN and Unicode mapping."
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    quality_results = []

    for record in result["records"]:

        q = character_quality(record)

        quality_results.append(q)

    good_count = sum(
        q["status"] == "GOOD"
        for q in quality_results
    )

    suspicious_count = (
        len(quality_results) -
        good_count
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Characters",
            len(result["records"])
        )

    with c2:
        st.metric(
            "Looks Reasonable",
            good_count
        )

    with c3:
        st.metric(
            "Suspicious Crops",
            suspicious_count
        )

    # --------------------------------------------------------
    # ALL EXTRACTED CHARACTERS
    # --------------------------------------------------------

    st.markdown(
        "### All Extracted Characters"
    )

    st.caption(
        "This is the most important debugging view. "
        "Each tile shows the actual crop sent toward recognition."
    )

    # Six characters per row.
    GRID_COLUMNS = 6

    records = result["records"]

    for start in range(
        0,
        len(records),
        GRID_COLUMNS
    ):

        row_records = records[
            start:start + GRID_COLUMNS
        ]

        columns = st.columns(
            GRID_COLUMNS
        )

        for local_index, record in enumerate(
            row_records
        ):

            global_number = (
                start +
                local_index +
                1
            )

            quality = character_quality(
                record
            )

            with columns[local_index]:

                # ------------------------------------------------
                # Character crop
                # ------------------------------------------------

                # ------------------------------------------------
                # ACTUAL SAVED EXTRACTED CHARACTER
                # ------------------------------------------------
                extracted_image = cv2.imread(
                    str(record["extracted_path"]),
                    cv2.IMREAD_GRAYSCALE
                )

                if extracted_image is not None:
                    st.image(
                        extracted_image,
                        caption=(
                            f"#{global_number}  "
                            f"L{record['line']}-"
                            f"C{record['char']}"
                        ),
                        width=110,
                    )
                else:
                    st.warning(
                        f"Could not load extracted image: "
                        f"{record['extracted_path']}"
                    )

                # ------------------------------------------------
                # Prediction
                # ------------------------------------------------

                st.caption(
                    f"Class: {record['label']}"
                )

                st.caption(
                    f"CNN: {record.get('cnn_label', record['label'])} "
                    f"({record['confidence'] * 100:.1f}%)"
                )

                st.caption(
                    f"Selected by: {record.get('decision_source', 'ALL_CLASS_CNN')}"
                )

                st.caption(
                    f"Recognition: "
                    f"{record.get('recognition_status', 'UNKNOWN')}"
                )

                # ------------------------------------------------
                # Unicode
                # ------------------------------------------------

                glyph = (
                    record["brahmi_glyph"]
                    or "N/A"
                )

                st.markdown(
                    f"**Unicode:** {glyph}"
                )

                # ------------------------------------------------
                # Diagnostic status
                # ------------------------------------------------

                st.caption(
                    f"{quality['status']} | "
                    f"{quality['width']}×"
                    f"{quality['height']}"
                )

    # --------------------------------------------------------
    # SELECT ONE CHARACTER
    # --------------------------------------------------------

    st.markdown(
        "### Detailed Character Inspection"
    )

    selected = st.number_input(
        "Select character number",
        min_value=1,
        max_value=len(records),
        value=1,
        step=1,
    )

    selected_index = (
        int(selected) - 1
    )

    record = records[
        selected_index
    ]

    quality = character_quality(
        record
    )

    # --------------------------------------------------------
    # CHARACTER INFORMATION
    # --------------------------------------------------------

    st.markdown(
        f"### Character #{record['global_index']}"
    )

    a, b, c = st.columns(3)

    # --------------------------------------------------------
    # ORIGINAL EXTRACTED CROP
    # --------------------------------------------------------

    with a:

        st.markdown(
            "**1. Extracted Character**"
        )

        extracted_image = cv2.imread(
            str(record["extracted_path"]),
            cv2.IMREAD_GRAYSCALE
        )

        if extracted_image is not None:
            st.image(
                extracted_image,
                width=220,
            )
        else:
            st.warning(
                "Extracted character image could not be loaded."
            )

        st.write(
            f"Line: {record['line']}"
        )

        st.write(
            f"Character: {record['char']}"
        )

        st.write(
            f"Bounding box: "
            f"({record['x']}, "
            f"{record['y']}, "
            f"{record['w']}, "
            f"{record['h']})"
        )

    # --------------------------------------------------------
    # CNN INPUT
    # --------------------------------------------------------

    with b:

        st.markdown(
            "**2. CNN Input**"
        )

        st.image(
            record["cnn"],
            width=220,
        )

        st.write(
            f"Input shape: "
            f"{record['cnn'].shape}"
        )

    # --------------------------------------------------------
    # RECOGNITION + MAPPING
    # --------------------------------------------------------

    with c:

        st.markdown(
            "**3. Recognition + Mapping**"
        )

        st.write(
            f"Final class: "
            f"`{record['label']}`"
        )

        st.write(
            f"CNN class: "
            f"`{record.get('cnn_label', record['label'])}`"
        )

        st.write(
            f"CNN confidence: "
            f"`{record['confidence'] * 100:.2f}%`"
        )

        st.write(
            f"Dataset class: "
            f"`{record.get('dataset_label') or 'N/A'}`"
        )

        st.write(
            f"Dataset similarity: "
            f"`{record.get('dataset_similarity', 0.0) * 100:.2f}%`"
        )

        st.write(
            f"Selected by: "
            f"`{record.get('decision_source', 'CNN')}`"
        )

        st.write(
            f"Recognition status: "
            f"`{record.get('recognition_status', recognition_status(record['confidence']))}`"
        )

        st.write(
            f"Brahmi Unicode: "
            f"`{record['brahmi_glyph'] or 'N/A'}`"
        )

        st.write(
            f"Unicode hex: "
            f"`{record['unicode_hex'] or 'N/A'}`"
        )

        st.write(
            f"Unicode name: "
            f"{record.get('unicode_character_name', '')}"
        )

        if record["brahmi_glyph"]:

            st.success(
                "Class → Unicode mapping exists"
            )

        else:

            st.error(
                "No Unicode mapping found for this class"
            )

    # --------------------------------------------------------
    # CHARACTER QUALITY
    # --------------------------------------------------------

    st.markdown(
        "### Segmentation Quality Diagnostics"
    )

    q1, q2, q3, q4 = st.columns(4)

    with q1:
        st.metric(
            "Width",
            quality["width"]
        )

    with q2:
        st.metric(
            "Height",
            quality["height"]
        )

    with q3:
        st.metric(
            "Aspect Ratio",
            f"{quality['aspect_ratio']:.2f}"
        )

    with q4:
        st.metric(
            "Foreground",
            f"{quality['foreground_ratio'] * 100:.1f}%"
        )

    if quality["status"] == "GOOD":

        st.success(
            "This crop passes the basic geometric diagnostics."
        )

    else:

        st.warning(
            f"Suspicious crop: {quality['status']}"
        )

    # --------------------------------------------------------
    # TOP-5 CNN PREDICTIONS
    # --------------------------------------------------------

    st.markdown(
        "### Top-5 CNN Predictions"
    )

    top5_table = pd.DataFrame([
        {
            "Rank": i + 1,
            "Class": item["label"],
            "Confidence": (
                f"{item['confidence'] * 100:.2f}%"
            ),
        }
        for i, item in enumerate(
            record["top5"]
        )
    ])

    st.dataframe(
        top5_table,
        hide_index=True,
        width="stretch",
    )

    # --------------------------------------------------------
    # MAPPING PIPELINE STATUS
    # --------------------------------------------------------

    st.markdown(
        "### Mapping Pipeline"
    )

    st.info(
        "The character was saved to the extracted folder, "
        "preprocessed to 64×64 grayscale, and the preprocessed "
        "image was passed to the all-class CNN. The predicted "
        "class is then mapped deterministically through the "
        "Brahmi Unicode CSV."
    )


# ============================================================
# 8. DIGITAL BRAHMI UNICODE
# ============================================================

st.subheader(
    "8. Digital Brahmi Unicode"
)

digital_lines = []

for line_no in sorted(
    set(
        r["line"]
        for r in result["records"]
    )
):

    current = [
        r
        for r in result["records"]
        if r["line"] == line_no
    ]

    # Left-to-right ordering
    current.sort(
        key=lambda r: (
            int(r.get("x", 0)),
            int(r.get("char", 0))
        )
    )

    line_text = "".join(
        r.get("brahmi_glyph", "")
        for r in current
        if r.get("brahmi_glyph", "")
    )

    digital_lines.append(
        line_text
    )

digital_text = "\n".join(
    digital_lines
)

st.text_area(
    "Digital Brahmi Unicode",
    value=digital_text,
    height=220,
)

if st.button(
    "Copy-ready Unicode Text",
    use_container_width=True
):

    st.code(
        digital_text,
        language=None
    )


# ============================================================
# 9. READABLE / TRANSLITERATED TEXT
# ============================================================

st.subheader(
    "9. Readable / Transliterated Text"
)

readable_text = build_readable_text(
    result["records"]
)

if readable_text:

    st.text_area(
        "Recognized Class Sequence",
        value=readable_text,
        height=180,
    )

else:

    st.warning(
        "No readable text was generated."
    )


# ============================================================
# 10. KANNADA TRANSLATION - CSV MAPPING
# ============================================================

st.subheader(
    "10. Kannada Translation"
)

st.write(
    "Convert each recognized English class name to its Kannada "
    "equivalent using the kannada_glyph column of brahmi_mapping.csv."
)

if readable_text:
    if st.button(
        "Convert to Kannada",
        type="primary",
        use_container_width=True,
    ):
        try:
            _, mapping = load_brahmi_mapping_csv()
            kannada_translation, missing_kannada = (
                build_kannada_translation(
                    result["records"],
                    mapping,
                )
            )

            if missing_kannada:
                st.warning(
                    "Kannada mapping is missing for: "
                    + ", ".join(missing_kannada)
                )

            st.success(
                "Kannada text generated directly from brahmi_mapping.csv."
            )

            st.text_area(
                "Kannada Translation",
                value=kannada_translation,
                height=180,
            )

            st.download_button(
                "Download Kannada Translation",
                data=kannada_translation,
                file_name="kannada_translation.txt",
                mime="text/plain",
                use_container_width=True,
            )

        except Exception as exc:
            st.error(
                "Could not generate Kannada translation from "
                f"brahmi_mapping.csv: {exc}"
            )
else:
    st.info(
        "Kannada translation will be available after character "
        "recognition generates the class sequence."
    )

