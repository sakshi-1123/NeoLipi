import os
import json
import cv2
import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from dline_segmentation_3_good_dup_2 import extract_line_strips
from crop_utils import standardize_stage4_crop
from translation_utils import translate_tamil_to_kannada, translate_batch_tamil_to_kannada

# Dataset Directory & Class Names Initialization
DATASET_DIR = r"D:\MAJOR_DATASET\character_dataset\tamil_inscription_dataset"
CLASS_NAMES = sorted(os.listdir(DATASET_DIR)) if os.path.exists(DATASET_DIR) else []

# =====================================================================
# STREAMLIT PAGE CONFIGURATION & STYLING
# =====================================================================
st.set_page_config(
    page_title="Stone Inscription OCR System",
    page_icon="📜",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Mildly zoom out slightly (92%) and add comfortable margin padding
st.markdown(
    """
    <style>
    .stApp {
        zoom: 92%;
    }
    .main .block-container {
        max-width: 92% !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

IMG_SIZE = 128
CHAR_IMG_SIZE = 64

# =====================================================================
# SCRIPT MAPPER & UNICODE MAPPINGS
# =====================================================================
DEVANAGARI_TO_UNICODE = {
    "a": "अ", "aa": "आ", "i": "इ", "ee": "ई", "ii": "ई", "u": "उ", "oo": "ऊ", "uu": "ऊ",
    "e": "ए", "ai": "ऐ", "o": "ओ", "au": "औ", "aaydham": "ः", "visarga": "ः",
    "ka": "क", "kha": "ख", "ga": "ग", "gha": "घ", "nga": "ङ", "cha": "च", "chha": "छ",
    "ja": "ज", "jha": "झ", "nya": "ञ", "ta": "त", "tha": "थ", "da": "द", "dha": "ध",
    "na": "न", "pa": "प", "pha": "फ", "ba": "ब", "bha": "भ", "ma": "म", "ya": "य",
    "ra": "र", "la": "ल", "va": "व", "sha": "श", "sa": "स", "ha": "ह",
    "k_mei": "क्", "ch_mei": "च्", "t_mei": "त्", "p_mei": "प्", "m_mei": "म्",
    "n_mei": "न्", "r_mei": "र्", "l_mei": "ल्", "v_mei": "व्", "y_mei": "य्",
    "kaa": "का", "ki": "कि", "kee": "की", "ku": "कु", "koo": "कू", "ke": "के",
    "kai": "कै", "ko": "को", "kau": "कौ"
}
BRAHMI_TO_UNICODE = {
    "a": "𑀁", "aa": "𑀂", "i": "𑀃", "ee": "𑀄", "u": "𑀅", "oo": "𑀆",
    "ka": "𑀓", "kha": "𑀔", "ga": "𑀕", "gha": "𑀖", "nga": "𑀗",
    "cha": "𑀘", "chha": "𑀙", "ja": "𑀚", "jha": "𑀛", "nja": "𑀜",
    "ta": "𑀝", "tha": "𑀞", "da": "𑀟", "dha": "𑀠", "nna": "𑀡",
    "tha_d": "𑀢", "thaa_d": "𑀣", "da_d": "𑀤", "dha_d": "𑀥", "na": "𑀦",
    "pa": "𑀧", "pha": "𑀨", "ba": "𑀩", "bha": "𑀪", "ma": "𑀫",
    "ya": "𑀬", "ra": "𑀭", "la": "𑀮", "va": "𑀯", "sha": "𑀰",
    "ssa": "𑀱", "sa": "𑀲", "ha": "𑀳", "lla": "𑀴"
}
LABEL_TO_UNICODE = {
    "a": "அ", "aa": "ஆ", "i": "இ", "ee": "ஈ", "u": "உ", "oo": "ஊ",
    "e": "எ", "ae": "ஏ", "ai": "ஐ", "o": "ஒ", "oa": "ஓ", "au": "ஔ",
    "aaydham": "ஃ", "sri": "ஸ்ரீ", "ksha": "க்ஷ",
    "ka": "க", "kaa": "கா", "ki": "கி", "kee": "கீ", "ku": "கு", "koo": "கூ", 
    "ke": "கெ", "kae": "கே", "kai": "கை", "ko": "கொ", "koa": "கோ", "kau": "கௌ", "k_mei": "க்",
    "nga": "ங", "ngaa": "ஙா", "ngi": "ஙி", "ngee": "ஙீ", "ngu": "ஙு", "ngoo": "ஙூ", 
    "nge": "ஙெ", "ngae": "ஙே", "ngai": "ஙை", "ngo": "ஙொ", "ngoa": "ஙோ", "ngau": "ஙௌ", "ng_mei": "ங்",
    "cha": "ச", "chaa": "சா", "chi": "சி", "chee": "சீ", "chu": "சு", "choo": "சூ", 
    "che": "செ", "chae": "சே", "chai": "சை", "cho": "சொ", "choa": "சோ", "chau": "சௌ", "ch_mei": "ச்",
    "nja": "ஞ", "njaa": "ஞா", "nji": "ஞி", "njee": "ஞீ", "nju": "ஞு", "njoo": "ஞூ", 
    "nje": "ஞெ", "njae": "ஞே", "njai": "ஞை", "njo": "ஞொ", "njoa": "ஞோ", "njau": "ஞௌ", "nj_mei": "ஞ்",
    "ta": "ட", "taa": "டா", "ti": "டி", "tee": "டீ", "tu": "டு", "too": "டூ", 
    "te": "டெ", "tae": "டே", "tai": "டை", "to": "டொ", "toa": "டோ", "tau": "டௌ", "t_mei": "ட்",
    "nna": "ண", "nnaa": "ணா", "nni": "ணி", "nnee": "ணீ", "nnu": "ணு", "nnoo": "ணூ", 
    "nne": "ணெ", "nnae": "ணே", "nnai": "ணை", "nno": "ணொ", "nnoa": "ணோ", "nnau": "ணௌ", "nn_mei": "ண்",
    "tha": "த", "thaa": "தா", "thi": "தி", "thee": "தீ", "thu": "து", "thoo": "தூ", 
    "the": "தெ", "thae": "தே", "thai": "தை", "tho": "தொ", "thoa": "தோ", "thau": "தௌ", "th_mei": "த்",
    "nha": "ந", "nhaa": "நா", "nhi": "நி", "nhee": "நீ", "nhu": "நு", "nhoo": "நூ", 
    "nhe": "நெ", "nhae": "நே", "nhai": "நை", "nho": "நொ", "nhoa": "நோ", "nhau": "நௌ", "nh_mei": "ந்",
    "pa": "ப", "paa": "பா", "pi": "பி", "pee": "பீ", "pu": "பு", "poo": "பூ", 
    "pe": "பெ", "pae": "பே", "pai": "பை", "po": "பொ", "poa": "போ", "pau": "பௌ", "p_mei": "ப்",
    "ma": "ம", "maa": "மா", "mi": "மி", "mee": "மீ", "mu": "மு", "moo": "மூ", 
    "me": "மெ", "mae": "மே", "mai": "மை", "mo": "மொ", "moa": "மோ", "mau": "மௌ", "m_mei": "ம்",
    "ya": "ய", "yaa": "யா", "yi": "யி", "yee": "யீ", "yu": "யு", "yoo": "யூ", 
    "ye": "யெ", "yae": "யே", "yai": "யை", "yo": "யொ", "yoa": "யோ", "yau": "யௌ", "y_mei": "ய்",
    "ra": "ர", "raa": "ரா", "ri": "ரி", "ree": "ரீ", "ru": "ரு", "roo": "ரூ", 
    "re": "ரெ", "rae": "ரே", "rai": "ரை", "ro": "ரொ", "roa": "ரோ", "rau": "ரௌ", "r_mei": "ர்",
    "la": "ல", "laa": "லா", "li": "லி", "lee": "லீ", "lu": "லு", "loo": "லூ", 
    "le": "லெ", "lae": "லே", "lai": "லை", "lo": "லொ", "loa": "லோ", "lau": "லௌ", "l_mei": "ல்",
    "va": "வ", "vaa": "வா", "vi": "வி", "vee": "வீ", "vu": "வு", "voo": "வூ", 
    "ve": "வெ", "vae": "வே", "vai": "வை", "vo": "வொ", "voa": "வோ", "vau": "வௌ", "v_mei": "வ்",
    "zha": "ழ", "zhaa": "ழா", "zhi": "ழி", "zhee": "ழீ", "zhu": "ழு", "zhoo": "ழூ", 
    "zhe": "ழெ", "zhae": "ழே", "zhai": "ழை", "zho": "ழொ", "zhoa": "ழோ", "zhau": "ழௌ", "zh_mei": "ழ்",
    "lha": "ள", "lhaa": "ளா", "lhi": "ளி", "lhee": "ளீ", "lhu": "ளு", "lhoo": "ளூ", 
    "lhe": "ளெ", "lhae": "ளே", "lhai": "ளை", "lho": "ளொ", "lhoa": "ளோ", "lhau": "ளௌ", "lh_mei": "ள்",
    "rha": "ற", "rhaa": "றா", "rhi": "றி", "rhee": "றீ", "rhu": "று", "rhoo": "றூ", 
    "rhe": "றெ", "rhae": "றே", "rhai": "றை", "rho": "றொ", "rhoa": "றோ", "rhau": "றௌ", "rh_mei": "ற்",
    "na": "ன", "naa": "னா", "ni": "னி", "nee": "னீ", "nu": "னு", "noo": "னூ", 
    "ne": "னெ", "nae": "னே", "nai": "னை", "no": "னொ", "noa": "னோ", "nau": "னௌ", "n_mei": "ன்",
    "ha": "ஹ", "ja": "ஜ", "sa": "ஸ", "sha": "ஷ", "la_retro": "ள"
}

ANCIENT_TAMIL_DICTIONARY = [
    "ஸ்வஸ்திஸ்ரீ", "கோயிலில்", "திருமலை", "கல்வெட்டு", "தானம்",
    "அரசன்", "சாசனம்", "பல்சுவை", "படைப்புகள்", "வேந்தன்",
    "மங்கலம்", "பொன்", "கொடை", "தேவதானம்", "பிரம்மதேயம்",
    "நிலம்", "சபை", "ஊரார்", "இறை", "கடமை", "நாயனார்", "வலந்பழப்பள"
]
from difflib import get_close_matches

# Known valid historical inscription dictionary/phrases
VALID_TAMIL_DICTIONARY = [
    "பல சக்திகள்",
    "பஹு சக்திகள்",
    "ஸ்ரீ ராஜராஜ",
    "சர்வ மானியம்"
]

def correct_linguistic_errors_normalized(raw_unicode_str, min_similarity=0.45):
    if not raw_unicode_str.strip():
        return raw_unicode_str
        
    words = raw_unicode_str.split()
    corrected_words = []
    
    for word in words:
        best_match = word
        best_sim = 0.0
        
        for candidate in ANCIENT_TAMIL_DICTIONARY:
            dist = levenshtein_distance(word, candidate)
            max_len = max(len(word), len(candidate))
            if max_len == 0:
                continue
            similarity = 1.0 - (dist / float(max_len))
            if similarity > best_sim:
                best_sim = similarity
                best_match = candidate
                
        corrected_words.append(best_match if best_sim >= min_similarity else word)
        
    return " ".join(corrected_words)

def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def correct_linguistic_errors_normalized(raw_unicode_str, min_similarity=0.50):
    if not raw_unicode_str.strip():
        return raw_unicode_str
        
    best_match = raw_unicode_str
    best_sim = 0.0
    
    for candidate in ANCIENT_TAMIL_DICTIONARY:
        dist = levenshtein_distance(raw_unicode_str, candidate)
        max_len = max(len(raw_unicode_str), len(candidate))
        if max_len == 0:
            continue
        similarity = 1.0 - (dist / float(max_len))
        if similarity > best_sim:
            best_sim = similarity
            best_match = candidate
            
    return best_match if best_sim >= min_similarity else raw_unicode_str

@st.cache_resource
def load_script_classifier():
    classifier_path = "script_classifier.h5"
    if os.path.exists(classifier_path):
        return tf.keras.models.load_model(classifier_path)
    return None

classifier_model = load_script_classifier()

# =====================================================================
# DIRECT CNN MODEL LOADER
# =====================================================================
@st.cache_resource
def load_direct_cnn_model(model_path="verification_model.h5", dataset_dir="train_verify_dataset"):
    if not os.path.exists(model_path):
        st.warning(f"Direct CNN model file '{model_path}' not found.")
        return None, []
    
    cnn_model = tf.keras.models.load_model(model_path)
    
    classes_json_path = "class_indices.json"
    if os.path.exists(classes_json_path):
        with open(classes_json_path, "r") as f:
            class_indices = json.load(f)
            classes = [k for k, v in sorted(class_indices.items(), key=lambda item: item[1])]
    elif os.path.exists(dataset_dir):
        classes = sorted([
            d for d in os.listdir(dataset_dir)
            if os.path.isdir(os.path.join(dataset_dir, d))
        ])
    else:
        classes = []
        
    return cnn_model, classes

def preprocess_morphology_stages(img_gray):
    smoothed = cv2.bilateralFilter(img_gray, d=5, sigmaColor=30, sigmaSpace=30)
    kernel_hat = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    blackhat = cv2.morphologyEx(smoothed, cv2.MORPH_BLACKHAT, kernel_hat)
    
    _, thresh = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    
    clean_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open)
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel_close)
    final_output = cv2.bitwise_not(clean_mask)
    
    return blackhat, thresh, clean_mask, final_output

import glob

def clear_line_stage4_crops(line_idx):
    """Deletes all existing crop files for a specific line in STAGE4_CROP_DIR."""
    pattern = os.path.join(STAGE4_CROP_DIR, f"line_{line_idx}_char_*.png")
    for filepath in glob.glob(pattern):
        try:
            os.remove(filepath)
        except Exception:
            pass

def classify_script(image):
    if classifier_model is None:
        return "Tamil", 50.0, {"Brahmi": 0.25, "Devanagari": 0.25, "Tamil": 0.50}

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 3)
    gray_resized = cv2.resize(blurred, (IMG_SIZE, IMG_SIZE)).astype("float32") / 255.0
    img_tensor = np.expand_dims(gray_resized, axis=(0, -1))

    preds = classifier_model.predict(img_tensor, verbose=0)[0]
    scores = {"Brahmi": float(preds[0]), "Devanagari": float(preds[1]), "Tamil": float(preds[2])}
    detected = max(scores, key=scores.get)
    return detected, scores[detected] * 100, scores

STAGE4_CROP_DIR = "stage4_segmented_crops"
os.makedirs(STAGE4_CROP_DIR, exist_ok=True)

def pad_and_resize_crop(crop_img, target_size=(128, 128)):
    if crop_img is None or crop_img.size == 0:
        return None
    h, w = crop_img.shape[:2]
    target_w, target_h = target_size
    
    scale = min(target_w / float(w), target_h / float(h))
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    
    resized = cv2.resize(crop_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    bg_color = int(np.median([crop_img[0,0], crop_img[0,-1], crop_img[-1,0], crop_img[-1,-1]]))
    
    if len(crop_img.shape) == 3:
        canvas = np.full((target_h, target_w, 3), bg_color, dtype=np.uint8)
    else:
        canvas = np.full((target_h, target_w), bg_color, dtype=np.uint8)
        
    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2
    
    canvas[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
    return canvas

def merge_overlapping_boxes_v2(boxes, overlap_thresh=0.25, max_gap=-2):
    if not boxes:
        return []
    
    boxes = sorted(boxes, key=lambda b: b[0])
    merged = [boxes[0]]
    
    for curr in boxes[1:]:
        prev_x, prev_y, prev_w, prev_h = merged[-1]
        curr_x, curr_y, curr_w, curr_h = curr
        
        prev_right = prev_x + prev_w
        curr_right = curr_x + curr_w
        
        overlap_x = max(0, min(prev_right, curr_right) - max(prev_x, curr_x))
        min_w = min(prev_w, curr_w)
        
        is_overlapping = min_w > 0 and (overlap_x / float(min_w)) > overlap_thresh
        is_close = (curr_x - prev_right) <= max_gap
        
        if is_overlapping or is_close:
            new_x = min(prev_x, curr_x)
            new_y = min(prev_y, curr_y)
            new_w = max(prev_right, curr_right) - new_x
            new_h = max(prev_y + prev_h, curr_y + curr_h) - new_y
            merged[-1] = (new_x, new_y, new_w, new_h)
        else:
            merged.append(curr)
            
    return merged

def segment_and_predict_characters(line_img, cnn_model, classes, selected_script="Tamil", line_idx=1):
    # Clear old crops for this line before saving new auto-segmented crops
    clear_line_stage4_crops(line_idx)

    if len(line_img.shape) == 3:
        gray = cv2.cvtColor(line_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = line_img.copy()

    if np.mean(gray) > 127:
        clean_mask = cv2.bitwise_not(gray)
        final_output = gray.copy()
    else:
        clean_mask = gray.copy()
        final_output = cv2.bitwise_not(gray)

    contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h_img, w_img = gray.shape
    raw_boxes = []

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = w / float(h)
        area = cv2.contourArea(c)
        
        roi = clean_mask[y:y+h, x:x+w]
        pixel_count = cv2.countNonZero(roi)

        if (h >= (h_img * 0.30) and h <= (h_img * 0.98) and 
            w >= 6 and aspect_ratio >= 0.12 and aspect_ratio <= 3.0 and
            area >= 20 and pixel_count >= 18):
            raw_boxes.append((x, y, w, h))

    boxes = merge_overlapping_boxes_v2(raw_boxes, overlap_thresh=0.40, max_gap=-3)
    segmented_bbox_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    character_crops = []
    predictions = []
    prediction_boxes = []

    PAD = 6

    for idx, (x, y, w, h) in enumerate(boxes, start=1):
        x1 = max(0, x - PAD)
        y1 = max(0, y - PAD)
        x2 = min(w_img, x + w + PAD)
        y2 = min(h_img, y + h + PAD)

        cv2.rectangle(segmented_bbox_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        disp_crop = final_output[y1:y2, x1:x2]
        if disp_crop.size == 0:
            continue

        standardized_crop = pad_and_resize_crop(disp_crop, target_size=(128, 128))
        if standardized_crop is None:
            continue

        character_crops.append(standardized_crop)

        if cnn_model is not None and len(classes) > 0:
            model_in = cv2.resize(standardized_crop, (CHAR_IMG_SIZE, CHAR_IMG_SIZE)).astype("float32") / 255.0

            input_shape = cnn_model.input_shape
            if len(model_in.shape) == 2 and input_shape[-1] == 3:
                model_in = cv2.cvtColor(model_in, cv2.COLOR_GRAY2BGR)

            if len(model_in.shape) == 2:
                tensor_in = np.expand_dims(model_in, axis=(0, -1))
            else:
                tensor_in = np.expand_dims(model_in, axis=0)

            raw_preds = cnn_model.predict(tensor_in, verbose=0)[0]
            top_idx = int(np.argmax(raw_preds))
            cnn_conf = float(raw_preds[top_idx])
            pred_folder = classes[top_idx]

            raw_label = pred_folder.split("_", 1)[1] if "_" in pred_folder else pred_folder

            if selected_script == "Brahmi":
                mapped_char = BRAHMI_TO_UNICODE.get(raw_label, raw_label)
            elif selected_script == "Devanagari":
                mapped_char = DEVANAGARI_TO_UNICODE.get(raw_label, raw_label)
            else:
                mapped_char = LABEL_TO_UNICODE.get(raw_label, raw_label)

            visual_conf_pct = round(cnn_conf * 100.0, 1)
        else:
            raw_label = "unknown"
            mapped_char = "?"
            cnn_conf = 0.0
            top_idx = 0
            visual_conf_pct = 0.0

        crop_filename = f"line_{line_idx}_char_{idx:02d}_{raw_label}.png"
        cv2.imwrite(os.path.join(STAGE4_CROP_DIR, crop_filename), standardized_crop)

        predictions.append((mapped_char, cnn_conf, visual_conf_pct, top_idx, raw_label))
        prediction_boxes.append((x, y, w, h))

    return segmented_bbox_img, character_crops, predictions, prediction_boxes

# ============================================================
# MANUAL CHARACTER SEGMENTATION HELPERS
# ============================================================

def _canvas_boxes_to_global(canvas_json, line_box, scale=1.0):
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

def process_manual_boxes_and_predict(line_img, drawn_boxes, cnn_model, classes, selected_script="Tamil", line_idx=1):
    clear_line_stage4_crops(line_idx)
    if len(line_img.shape) == 3:
        gray = cv2.cvtColor(line_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = line_img.copy()

    final_output = gray.copy() if np.mean(gray) > 127 else cv2.bitwise_not(gray)
    h_img, w_img = gray.shape
    segmented_bbox_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    character_crops = []
    predictions = []
    prediction_boxes = []

    sorted_boxes = sorted(drawn_boxes, key=lambda b: b[0])
    PAD = 4

    for idx, (x, y, w, h) in enumerate(sorted_boxes, start=1):
        if w <= 0 or h <= 0:
            continue

        x1 = max(0, x - PAD)
        y1 = max(0, y - PAD)
        x2 = min(w_img, x + w + PAD)
        y2 = min(h_img, y + h + PAD)

        cv2.rectangle(segmented_bbox_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        disp_crop = final_output[y1:y2, x1:x2]
        if disp_crop.size == 0:
            continue

        standardized_crop = pad_and_resize_crop(disp_crop, target_size=(128, 128))
        if standardized_crop is None:
            continue

        character_crops.append(standardized_crop)

        if cnn_model is not None and len(classes) > 0:
            model_in = cv2.resize(standardized_crop, (CHAR_IMG_SIZE, CHAR_IMG_SIZE)).astype("float32") / 255.0

            input_shape = cnn_model.input_shape
            if len(model_in.shape) == 2 and input_shape[-1] == 3:
                model_in = cv2.cvtColor(model_in, cv2.COLOR_GRAY2BGR)

            if len(model_in.shape) == 2:
                tensor_in = np.expand_dims(model_in, axis=(0, -1))
            else:
                tensor_in = np.expand_dims(model_in, axis=0)

            raw_preds = cnn_model.predict(tensor_in, verbose=0)[0]
            top_idx = int(np.argmax(raw_preds))
            cnn_conf = float(raw_preds[top_idx])
            pred_folder = classes[top_idx]

            raw_label = pred_folder.split("_", 1)[1] if "_" in pred_folder else pred_folder

            if selected_script == "Brahmi":
                mapped_char = BRAHMI_TO_UNICODE.get(raw_label, raw_label)
            elif selected_script == "Devanagari":
                mapped_char = DEVANAGARI_TO_UNICODE.get(raw_label, raw_label)
            else:
                mapped_char = LABEL_TO_UNICODE.get(raw_label, raw_label)

            visual_conf_pct = round(cnn_conf * 100.0, 1)
        else:
            raw_label = "unknown"
            mapped_char = "?"
            cnn_conf = 0.0
            top_idx = 0
            visual_conf_pct = 0.0

        # Save crops directly into Stage 4 directory
        crop_filename = f"line_{line_idx}_char_{idx:02d}_{raw_label}.png"
        cv2.imwrite(os.path.join(STAGE4_CROP_DIR, crop_filename), standardized_crop)

        predictions.append((mapped_char, cnn_conf, visual_conf_pct, top_idx, raw_label))
        prediction_boxes.append((x, y, w, h))

    return segmented_bbox_img, character_crops, predictions, prediction_boxes

def extract_manual_boxes_from_canvas(json_data, scale_x=1.0, scale_y=1.0):
    """Extract bounding boxes from st_canvas and scale back to original image pixel coords."""
    boxes = []
    if not json_data:
        return boxes

    if isinstance(json_data, str):
        try:
            json_data = json.loads(json_data)
        except Exception:
            return boxes

    if not isinstance(json_data, dict) or "objects" not in json_data:
        return boxes

    for obj in json_data.get("objects", []):
        obj_type = str(obj.get("type", "")).lower()
        if obj_type in ["rect", "rectangle"]:
            left = float(obj.get("left", 0.0))
            top = float(obj.get("top", 0.0))
            width = float(obj.get("width", 0.0))
            height = float(obj.get("height", 0.0))
            sx = float(obj.get("scaleX", 1.0))
            sy = float(obj.get("scaleY", 1.0))

            actual_w = abs(width * sx)
            actual_h = abs(height * sy)
            actual_x = left if sx >= 0 else left - actual_w
            actual_y = top if sy >= 0 else top - actual_h

            # Scale coordinates from canvas display size back to original image size
            orig_x = int(round(actual_x * scale_x))
            orig_y = int(round(actual_y * scale_y))
            orig_w = int(round(actual_w * scale_x))
            orig_h = int(round(actual_h * scale_y))

            if orig_w > 1 and orig_h > 1:
                boxes.append((orig_x, orig_y, orig_w, orig_h))

    boxes.sort(key=lambda b: b[0])
    return boxes

# =====================================================================
# STREAMLIT UI LAYOUT & CONTROL FLOW
# =====================================================================
st.title("📜 Stone Inscription OCR & Multi-Script Recognition System")
st.markdown("Upload a stone inscription image to run script classification, morphological preprocessing, line segmentation, character detection, direct CNN inference, and translation.")

with st.sidebar:
    st.header("⚙️ System Control Panel")
    segmentation_mode = st.radio(
        "Select Character Bounding Box Mode:",
        options=["Automatic Contour Detection", "Manual Canvas Bounding Box"],
        index=0
    )
    st.write("---")
    st.info("Supported Scripts:\n- **Brahmi**\n- **Devanagari**\n- **Tamil**")

uploaded_file = st.file_uploader("Upload Stone Inscription Image", type=["jpg", "jpeg", "png", "bmp"], key="stone_inscription_uploader")

if uploaded_file is not None:
    # Clear all stage 4 crops from previous image runs
    for f in glob.glob(os.path.join(STAGE4_CROP_DIR, "*.png")):
        os.remove(f)
    image = Image.open(uploaded_file)
    col_input, col_process = st.columns([1, 1], gap="large")

    with col_input:
        st.subheader("Original Uploaded Inscription")
        st.image(image, use_container_width=True)

    img = np.array(image)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    with col_process:
        st.subheader("Script Classification & Target Routing")
        detected_script, confidence, scores = classify_script(img)

        m1, m2, m3 = st.columns(3)
        m1.metric("Brahmi", f"{scores['Brahmi']*100:.1f}%")
        m2.metric("Devanagari", f"{scores['Devanagari']*100:.1f}%")
        m3.metric("Tamil", f"{scores['Tamil']*100:.1f}%")

        selected_script = st.selectbox(
            "Classified Script:",
            options=["Tamil", "Brahmi", "Devanagari"],
            index=["Tamil", "Brahmi", "Devanagari"].index(detected_script)
        )
    st.write("")
    run_btn = st.button("Process & Route Full Pipeline", type="primary", use_container_width=True)
    st.markdown("---")

    cnn_model, classes = load_direct_cnn_model("verification_model.h5", "train_verify_dataset")
    
    full_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    full_blackhat, full_thresh, full_clean_mask, full_final_output = preprocess_morphology_stages(full_gray)

    st.subheader("1. Full-Image Morphological Preprocessing Stages")
    p1, p2, p3, p4 = st.columns(4)
    p1.image(full_gray, caption="Original Stone Image", use_container_width=True)
    p2.image(full_blackhat, caption="Background Removed (Blackhat)", use_container_width=True)
    p3.image(full_thresh, caption="Binary Mask", use_container_width=True)
    p4.image(full_final_output, caption="Clean Inverted Mask", use_container_width=True)

    st.subheader("2. Line Detection & Segmentation (From Clean Mask)")
    line_crops, debug_img, _ = extract_line_strips(full_thresh, is_preprocessed=True)        
    st.image(debug_img, caption="HPP Line Segmentation Projection on Clean Inverted Mask", use_container_width=True)
    st.write(f"**Total Line Strips Isolated:** `{len(line_crops)}`")

    if len(line_crops) == 0:
        st.warning("⚠️ No distinct line crops isolated. Please adjust line segmentation parameters.")
    else:
        st.markdown("---")
        st.subheader("3. Character Segmentation & Recognition Pipeline")

        full_raw_text = []
        full_corrected_text = []
        export_data = []

        for line_item in line_crops:
            line_idx = line_item["line_idx"]
            crop_image = line_item["image"]

            if crop_image is None or crop_image.size == 0:
                continue

            st.markdown(f"### 📍 Line #{line_idx} Execution Pipeline")

            # =========================================================
            # STAGE 3: UNIFIED CHARACTER SEGMENTATION (MANUAL OR AUTO)
            # =========================================================
            if segmentation_mode == "Manual Canvas Bounding Box":
                st.markdown("**Stage 3: Character Segmentation (Manual Draw)**")
                st.caption("Click and drag green rectangles around each character directly on the line strip below.")

                orig_h, orig_w = crop_image.shape[:2]
                
                # Canvas scaled to 900px wide for easy drawing
                target_canvas_w = 900
                target_canvas_h = max(140, int(target_canvas_w * (orig_h / float(orig_w))))

                bg_display_img = cv2.resize(crop_image, (target_canvas_w, target_canvas_h), interpolation=cv2.INTER_CUBIC)
                pil_bg_img = Image.fromarray(cv2.cvtColor(bg_display_img, cv2.COLOR_BGR2RGB))

                scale_x = orig_w / float(target_canvas_w)
                scale_y = orig_h / float(target_canvas_h)

                file_id = f"{uploaded_file.name}_{uploaded_file.size}"

                # Single Interactive Canvas Stage (Acts as drawing canvas & visual preview)
                canvas_result = st_canvas(
                    fill_color="rgba(0, 255, 0, 0.25)",
                    stroke_width=2,
                    stroke_color="#00FF00",
                    background_image=pil_bg_img,
                    update_streamlit=True,
                    height=target_canvas_h,
                    width=target_canvas_w,
                    drawing_mode="rect",
                    key=f"canvas_{file_id}_line_{line_idx}"
                )

                manual_boxes = extract_manual_boxes_from_canvas(
                    canvas_result.json_data, 
                    scale_x=scale_x, 
                    scale_y=scale_y
                )

                if manual_boxes:
                    _, character_crops, predictions, prediction_boxes = process_manual_boxes_and_predict(
                        crop_image,
                        manual_boxes,
                        cnn_model=cnn_model,
                        classes=classes,
                        selected_script=selected_script,
                        line_idx=line_idx
                    )
                else:
                    st.info("👈 Draw rectangles over each character directly on the canvas above.")
                    character_crops, predictions = [], []
                    prediction_boxes = []

            else:
                # Automatic Contour Detection Mode
                st.markdown("**Stage 3: Character Segmentation (Automatic Contour)**")
                seg_bbox_img, character_crops, predictions, prediction_boxes = segment_and_predict_characters(
                    crop_image,
                    cnn_model=cnn_model,
                    classes=classes,
                    selected_script=selected_script,
                    line_idx=line_idx
                )
                st.image(seg_bbox_img, caption="Segmented Line Strip with Green Bounding Boxes", use_container_width=True)

            # =========================================================
            # STAGE 4: DIRECT CNN INFERENCE & UNICODE OUTPUT
            # =========================================================
            if character_crops:
                st.markdown("**Stage 4: Direct CNN Inference & Unicode Output**")

                # Render all characters side-by-side in a single horizontal row
                cols = st.columns(len(character_crops))

                for col, crop, pred in zip(cols, character_crops, predictions):
                    mapped_char, cnn_conf, visual_conf_pct, pred_idx, raw_label = pred
                    with col:
                        st.image(crop, use_container_width=True)
                        st.markdown(f"##### **{mapped_char}**")
                        st.caption(f"Idx: `{pred_idx}` | `{raw_label}`")
                        st.caption(f"Conf: {cnn_conf * 100:.1f}%")
            else:
                st.warning("No valid characters detected for this line.")

            if predictions:
                raw_chars_with_spaces = []
                gap_threshold = 25

                for i, prediction in enumerate(predictions):
                    raw_chars_with_spaces.append(prediction[0])
                    if i < len(prediction_boxes) - 1:
                        current_x2 = prediction_boxes[i][0] + prediction_boxes[i][2]
                        next_x1 = prediction_boxes[i + 1][0]
                        if next_x1 - current_x2 > gap_threshold:
                            raw_chars_with_spaces.append(" ")

                line_raw_chars = [p[0] for p in predictions]
                line_cnn_confs = [p[1] for p in predictions]

                raw_line_str = "".join(raw_chars_with_spaces)
                corrected_line_str = correct_linguistic_errors_normalized(raw_line_str)

                full_raw_text.append(raw_line_str)
                full_corrected_text.append(corrected_line_str)

                line_avg_conf = np.mean(line_cnn_confs) * 100 if line_cnn_confs else 0.0
                export_data.append({
                    "Line Index": line_idx,
                    "Raw OCR Output": raw_line_str,
                    "Corrected Text (Tamil)": corrected_line_str,
                    "Character Count": len(line_raw_chars),
                    "Avg Confidence (%)": round(line_avg_conf, 2)
                })

            st.markdown("---")

        full_kannada_text = translate_batch_tamil_to_kannada(full_corrected_text)

        for idx, (raw, corr, kan) in enumerate(zip(full_raw_text, full_corrected_text, full_kannada_text), start=1):
            st.markdown(f"**Stage 5: Line #{idx} Output & Translation**")
            st.info(
                f"**Original (CNN Output):** `{raw}`\n\n"
                f"**Corrected Text (Tamil):** `{corr}`\n\n"
                f"**Kannada Translation:** `{kan}`"
            )

        st.subheader("📜 Final Inscription Output Stream & Translation")
        c_raw, c_corr, c_kan = st.columns(3)

        with c_raw:
            st.text_area("Raw Predictions:", value="\n".join(full_raw_text), height=180)

        with c_corr:
            st.text_area("Corrected Output (Tamil):", value="\n".join(full_corrected_text), height=180)

        with c_kan:
            st.text_area("Kannada Translation:", value="\n".join(full_kannada_text), height=180)