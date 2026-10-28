import cv2
import numpy as np
import os
import sys


# ============================================================
# NeoLipi - Brahmi Stone Inscription Binarization
# ============================================================
#
# Designed specifically for:
#   - stone inscriptions
#   - dark engraved strokes
#   - uneven stone illumination
#   - stone texture
#   - shallow/faint characters
#
# Pipeline:
#
# Original
#    ↓
# Stone ROI extraction
#    ↓
# Illumination normalization
#    ↓
# CLAHE
#    ↓
# Multi-scale black-hat stroke extraction
#    ↓
# Response fusion
#    ↓
# Percentile threshold
#    ↓
# Small-gap stroke reconstruction
#    ↓
# Small-noise removal
#    ↓
# CLEAN BINARY MASK
#
# WHITE = inscription strokes
# BLACK = stone/background
# ============================================================


OUTPUT_DIR = "outputs"


# ------------------------------------------------------------
# PARAMETERS
# ------------------------------------------------------------

# Main threshold.
#
# Higher:
#   fewer/noisier strokes
#
# Lower:
#   more strokes but more stone texture
#
# 94 works well for the Brahmi test image we are currently
# using.
BLACKHAT_PERCENTILE = 92.0


# Black-hat scales.
#
# Multiple scales allow the system to capture:
#   thin strokes
#   medium strokes
#   thicker engraved regions
BLACKHAT_SCALES = [15, 21, 31]


# Small gap repair.
STROKE_CLOSE_KERNEL = 3


# Remove components smaller than this.
MIN_COMPONENT_PIXELS = 15


# ------------------------------------------------------------
# DIRECTORY
# ------------------------------------------------------------

def ensure_output_dir():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )


# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

def save(name, image):

    path = os.path.join(
        OUTPUT_DIR,
        name
    )

    cv2.imwrite(
        path,
        image
    )

    print(
        f"[SAVED] {path}"
    )


# ------------------------------------------------------------
# LOAD
# ------------------------------------------------------------

def load_image(path):

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"\nImage does not exist:\n{path}"
        )

    image = cv2.imread(path)

    if image is None:

        raise ValueError(
            f"\nOpenCV could not read:\n{path}"
        )

    return image


# ------------------------------------------------------------
# FIND STONE REGION
# ------------------------------------------------------------

def find_stone_region(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Background of the supplied Brahmi image is nearly white.
    #
    # This gives us an approximate stone mask.
    stone_mask = np.where(
        gray < 245,
        255,
        0
    ).astype(np.uint8)

    # Remove tiny noise.
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    stone_mask = cv2.morphologyEx(
        stone_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    stone_mask = cv2.morphologyEx(
        stone_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    # Largest connected region = stone.
    count, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            stone_mask,
            connectivity=8
        )
    )

    if count <= 1:

        return (
            image,
            np.ones(
                gray.shape,
                dtype=np.uint8
            ) * 255,
            (0, 0, image.shape[1], image.shape[0])
        )

    largest = 1 + np.argmax(
        stats[1:, cv2.CC_STAT_AREA]
    )

    stone_region = np.where(
        labels == largest,
        255,
        0
    ).astype(np.uint8)

    x = stats[
        largest,
        cv2.CC_STAT_LEFT
    ]

    y = stats[
        largest,
        cv2.CC_STAT_TOP
    ]

    w = stats[
        largest,
        cv2.CC_STAT_WIDTH
    ]

    h = stats[
        largest,
        cv2.CC_STAT_HEIGHT
    ]

    return (
        image,
        stone_region,
        (x, y, w, h)
    )


# ------------------------------------------------------------
# GRAYSCALE
# ------------------------------------------------------------

def to_gray(image):

    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )


# ------------------------------------------------------------
# ILLUMINATION NORMALIZATION
# ------------------------------------------------------------

def normalize_illumination(gray):

    # Estimate slowly varying stone illumination.
    background = cv2.GaussianBlur(
        gray,
        (0, 0),
        sigmaX=18,
        sigmaY=18
    )

    background = np.maximum(
        background,
        1
    )

    normalized = cv2.divide(
        gray,
        background,
        scale=180
    )

    normalized = cv2.normalize(
        normalized,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    return normalized.astype(
        np.uint8
    )


# ------------------------------------------------------------
# CLAHE
# ------------------------------------------------------------

def enhance_local_contrast(gray):

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(
        gray
    )

    return enhanced


# ------------------------------------------------------------
# BLACK-HAT STROKE EXTRACTION
# ------------------------------------------------------------

def blackhat_response(gray, kernel_size):

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            kernel_size,
            kernel_size
        )
    )

    # Black-hat:
    #
    # closing(image) - image
    #
    # Dark local structures are enhanced.
    response = cv2.morphologyEx(
        gray,
        cv2.MORPH_BLACKHAT,
        kernel
    )

    return response


# ------------------------------------------------------------
# NORMALIZE RESPONSE
# ------------------------------------------------------------

def normalize_response(response):

    # Percentile normalization prevents a few extreme
    # pixels from dominating the image.

    low = np.percentile(
        response,
        1
    )

    high = np.percentile(
        response,
        99
    )

    if high <= low:

        return np.zeros_like(
            response
        )

    normalized = (
        response.astype(
            np.float32
        ) - low
    ) / (
        high - low
    )

    normalized *= 255.0

    normalized = np.clip(
        normalized,
        0,
        255
    )

    return normalized.astype(
        np.uint8
    )


# ------------------------------------------------------------
# MULTI-SCALE FUSION
# ------------------------------------------------------------

def multi_scale_blackhat(gray):

    responses = []

    for scale in BLACKHAT_SCALES:

        print(
            f"[INFO] Black-hat scale: {scale}"
        )

        response = blackhat_response(
            gray,
            scale
        )

        response = normalize_response(
            response
        )

        save(
            f"blackhat_{scale}.png",
            response
        )

        responses.append(
            response.astype(
                np.float32
            ) / 255.0
        )

    # Average instead of simple max.
    #
    # MAX tends to bring back stone texture.
    #
    # Average requires a stroke to have support across
    # multiple scales, which is more conservative.
    fused = np.mean(
        responses,
        axis=0
    )

    fused = np.clip(
        fused * 255.0,
        0,
        255
    ).astype(
        np.uint8
    )

    return fused


# ------------------------------------------------------------
# CREATE INITIAL MASK
# ------------------------------------------------------------

def threshold_response(
    response,
    stone_mask,
    percentile
):

    # Only consider pixels inside the stone.
    valid_pixels = response[
        stone_mask > 0
    ]

    if len(valid_pixels) == 0:

        raise ValueError(
            "Stone region could not be detected."
        )

    threshold = np.percentile(
        valid_pixels,
        percentile
    )

    print(
        f"[INFO] Black-hat threshold: "
        f"{threshold:.2f}"
    )

    binary = np.zeros_like(
        response,
        dtype=np.uint8
    )

    binary[
        response >= threshold
    ] = 255

    # Never allow outside-stone pixels.
    binary[
        stone_mask == 0
    ] = 0

    return binary


# ------------------------------------------------------------
# STROKE RECONSTRUCTION
# ------------------------------------------------------------

def reconnect_strokes(binary):

    # First close very small horizontal gaps.
    horizontal = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (3, 1)
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        horizontal
    )

    # Then small vertical gaps.
    vertical = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (1, 3)
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        vertical
    )

    # Small isotropic closing.
    ellipse = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            STROKE_CLOSE_KERNEL,
            STROKE_CLOSE_KERNEL
        )
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        ellipse
    )

    return binary


# ------------------------------------------------------------
# REMOVE SMALL COMPONENTS
# ------------------------------------------------------------

def remove_small_components(
    binary,
    minimum_area=MIN_COMPONENT_PIXELS
):

    count, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            binary,
            connectivity=8
        )
    )

    result = np.zeros_like(
        binary
    )

    kept = 0
    removed = 0

    for i in range(1, count):

        area = stats[
            i,
            cv2.CC_STAT_AREA
        ]

        if area >= minimum_area:

            result[
                labels == i
            ] = 255

            kept += 1

        else:

            removed += 1

    print(
        f"[INFO] Components kept: {kept}"
    )

    print(
        f"[INFO] Tiny components removed: {removed}"
    )

    return result


# ------------------------------------------------------------
# REMOVE VERY THIN BORDER ARTIFACTS
# ------------------------------------------------------------

def remove_border_artifacts(
    binary,
    stone_mask
):

    # Keep only pixels sufficiently inside the stone region.
    #
    # This prevents the outer stone boundary from becoming
    # a large false character.

    distance = cv2.distanceTransform(
        stone_mask,
        cv2.DIST_L2,
        5
    )

    # Do NOT use a large margin.
    #
    # 3 pixels is deliberately conservative.
    safe_region = np.where(
        distance >= 3,
        255,
        0
    ).astype(np.uint8)

    result = cv2.bitwise_and(
        binary,
        safe_region
    )

    return result


# ------------------------------------------------------------
# FILL SMALL INTERNAL GAPS
# ------------------------------------------------------------

def gentle_fill(binary):

    # Very small closing only.
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2, 2)
    )

    return cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel
    )


# ------------------------------------------------------------
# FINAL CLEAN BINARY
# ------------------------------------------------------------

def make_final_binary(
    fused_response,
    stone_mask
):

    binary = threshold_response(
        fused_response,
        stone_mask,
        BLACKHAT_PERCENTILE
    )

    save(
        "04_initial_binary.png",
        binary
    )

    # Reconnect small breaks.
    binary = reconnect_strokes(
        binary
    )

    save(
        "05_stroke_reconnected.png",
        binary
    )

    # Remove only tiny speckles.
    binary = remove_small_components(
        binary
    )

    save(
        "06_noise_removed.png",
        binary
    )

    # Remove outer stone boundary artifacts.
    binary = remove_border_artifacts(
        binary,
        stone_mask
    )

    save(
        "07_border_cleaned.png",
        binary
    )

    # Gentle final filling.
    binary = gentle_fill(
        binary
    )

    save(
        "08_FINAL_BINARY.png",
        binary
    )

    return binary


# ------------------------------------------------------------
# CREATE DEBUG OVERLAY
# ------------------------------------------------------------

def create_overlay(
    original,
    binary
):

    overlay = original.copy()

    # Red/green visualization of detected strokes.
    # Green = detected inscription.
    overlay[
        binary > 0
    ] = (
        0,
        255,
        0
    )

    blended = cv2.addWeighted(
        original,
        0.65,
        overlay,
        0.35,
        0
    )

    return blended


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def process(image_path):

    ensure_output_dir()

    print(
        "\n================================================"
    )

    print(
        "NeoLipi Brahmi Stone Binarization"
    )

    print(
        "================================================\n"
    )

    print(
        f"[INFO] Input:\n{image_path}\n"
    )

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    image = load_image(
        image_path
    )

    save(
        "01_original.png",
        image
    )

    # --------------------------------------------------------
    # STONE REGION
    # --------------------------------------------------------

    _, stone_mask, bbox = (
        find_stone_region(
            image
        )
    )

    print(
        f"[INFO] Stone bounding box: {bbox}"
    )

    save(
        "02_stone_mask.png",
        stone_mask
    )

    # --------------------------------------------------------
    # GRAYSCALE
    # --------------------------------------------------------

    gray = to_gray(
        image
    )

    # Keep the outside background neutral.
    gray_for_processing = gray.copy()

    # --------------------------------------------------------
    # ILLUMINATION NORMALIZATION
    # --------------------------------------------------------

    normalized = normalize_illumination(
        gray_for_processing
    )

    save(
        "03_illumination_normalized.png",
        normalized
    )

    # --------------------------------------------------------
    # LOCAL CONTRAST
    # --------------------------------------------------------

    enhanced = enhance_local_contrast(
        normalized
    )

    save(
        "03b_contrast.png",
        enhanced
    )

    # --------------------------------------------------------
    # MULTI-SCALE BLACK-HAT
    # --------------------------------------------------------

    fused = multi_scale_blackhat(
        enhanced
    )

    save(
        "03c_multiscale_fused.png",
        fused
    )

    # --------------------------------------------------------
    # FINAL BINARY
    # --------------------------------------------------------

    final_binary = make_final_binary(
        fused,
        stone_mask
    )

    # --------------------------------------------------------
    # OVERLAY
    # --------------------------------------------------------

    overlay = create_overlay(
        image,
        final_binary
    )

    save(
        "09_detection_overlay.png",
        overlay
    )

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    stone_pixels = np.count_nonzero(
        stone_mask
    )

    text_pixels = np.count_nonzero(
        final_binary
    )

    density = (
        text_pixels /
        max(stone_pixels, 1)
    )

    print(
        "\n================================================"
    )

    print(
        "FINAL RESULT"
    )

    print(
        "================================================"
    )

    print(
        f"Stone pixels : {stone_pixels}"
    )

    print(
        f"Text pixels  : {text_pixels}"
    )

    print(
        f"Text density : {density:.4f}"
    )

    print(
        "\nFINAL BINARY:"
    )

    print(
        os.path.abspath(
            os.path.join(
                OUTPUT_DIR,
                "08_FINAL_BINARY.png"
            )
        )
    )

    print(
        "\n================================================"
    )

    return final_binary


# ------------------------------------------------------------
# COMMAND LINE
# ------------------------------------------------------------

if __name__ == "__main__":

    if len(sys.argv) < 2:

        print(
            '\nUsage:\n'
            'python binarization.py '
            '"D:\\path\\to\\brahmi.png"\n'
        )

        sys.exit(1)

    image_path = sys.argv[1]

    process(
        image_path
    )