import cv2
import numpy as np
from scipy.signal import find_peaks


def refine_vatteluttu_strokes(binary_img, min_noise_area=15):
    """
    Light refinement that preserves stroke thickness while clearing surface noise.
    """
    if len(binary_img.shape) == 3:
        binary_img = cv2.cvtColor(binary_img, cv2.COLOR_BGR2GRAY)

    # Ensure white text (255) on black background (0)
    if np.mean(binary_img) > 127:
        binary_img = cv2.bitwise_not(binary_img)

    # Connected component noise filtering
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_img, connectivity=8)
    filtered = np.zeros_like(binary_img)

    for i in range(1, num_labels):  # Skip background (0)
        if stats[i, cv2.CC_STAT_AREA] >= min_noise_area:
            filtered[labels == i] = 255

    return filtered


def merge_overlapping_line_bounds(line_bounds, overlap_threshold=0.30):
    """
    Merges bounding boxes that physically overlap vertically or sit too close,
    preventing duplicate overlapping line detections.
    """
    if not line_bounds:
        return []

    sorted_bounds = sorted(line_bounds, key=lambda b: b[0])
    merged = [list(sorted_bounds[0])]

    for current_y1, current_y2 in sorted_bounds[1:]:
        prev_y1, prev_y2 = merged[-1]
        prev_height = prev_y2 - prev_y1
        curr_height = current_y2 - current_y1

        overlap = max(0, min(prev_y2, current_y2) - max(prev_y1, current_y1))
        min_height = min(prev_height, curr_height)

        if overlap / float(min_height) > overlap_threshold or current_y1 < prev_y2 - (0.3 * prev_height):
            merged[-1][1] = max(prev_y2, current_y2)
        else:
            merged.append([current_y1, current_y2])

    return [(b[0], b[1]) for b in merged]


def standardize_and_clean_image(image_bgr):
    """
    Standardizes binarization and applies dynamic border zeroing.
    """
    if len(image_bgr.shape) == 3:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_bgr.copy()

    h, w = gray.shape

    # Binarization check
    unique_vals = np.unique(gray)
    if len(unique_vals) <= 10 or (np.mean(gray > 200) + np.mean(gray < 50) > 0.85):
        binary = cv2.bitwise_not(gray) if np.mean(gray) > 127 else gray.copy()
    else:
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 51, 15
        )

    # Dynamic border zeroing based on dimensions
    margin_x = max(4, int(w * 0.025))
    margin_y = max(4, int(h * 0.020))

    binary[:, :margin_x] = 0
    binary[:, -margin_x:] = 0
    binary[:margin_y, :] = 0
    binary[-margin_y:, :] = 0

    # Refine strokes for clean Vatteluttu text
    min_area = max(15, int((h * w) * 0.0001))
    enhanced_binary = refine_vatteluttu_strokes(binary, min_noise_area=min_area)

    return gray, enhanced_binary


def extract_line_strips(binary_mask, line_padding=3, is_preprocessed=True, **kwargs):
    """
    Extracts line crops and tight bounding boxes from stone inscription images/masks.
    Uses Horizontal Projection Profile (HPP) peak detection and VPP horizontal trimming.
    """
    if len(binary_mask.shape) == 3:
        binary_mask = cv2.cvtColor(binary_mask, cv2.COLOR_BGR2GRAY)

    if not is_preprocessed:
        gray, binary = standardize_and_clean_image(binary_mask)
    else:
        gray = binary_mask.copy()
        binary = cv2.bitwise_not(binary_mask) if np.mean(binary_mask) > 127 else binary_mask.copy()

    h, w = binary.shape

    # Apply horizontal dilation to group characters into cohesive lines
    margin_x = max(5, int(w * 0.03))
    inner_cleaned = binary[:, margin_x:w - margin_x] if w > (2 * margin_x) else binary
    morph_w = max(14, int(w * 0.04))
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_w, 1))
    dilated_inner = cv2.dilate(inner_cleaned, kernel_h, iterations=2)

    # Smooth Horizontal Projection Profile (HPP)
    hpp = np.sum(dilated_inner, axis=1) / 255.0
    smooth_w = max(5, int(h * 0.025))
    if smooth_w % 2 == 0:
        smooth_w += 1
    hpp_smoothed = np.convolve(hpp, np.ones(smooth_w) / smooth_w, mode='same')

    # Peak detection for line segmentation
    min_line_height = max(14, int(h * 0.045))
    max_hpp = np.max(hpp_smoothed) if len(hpp_smoothed) > 0 else 0

    if max_hpp > 0:
        peaks, _ = find_peaks(
            hpp_smoothed, 
            distance=min_line_height, 
            prominence=max_hpp * 0.06,
            height=max_hpp * 0.02
        )
    else:
        peaks = []

    raw_line_bounds = []

    if len(peaks) <= 1:
        non_zero_y = np.where(hpp_smoothed > (max_hpp * 0.04 if max_hpp > 0 else 0))[0]
        if len(non_zero_y) > 0:
            y_min = max(0, non_zero_y[0] - line_padding)
            y_max = min(h, non_zero_y[-1] + line_padding)
            if (y_max - y_min) >= min_line_height:
                raw_line_bounds.append((y_min, y_max))
    else:
        thresh = max_hpp * 0.02
        top_y = peaks[0]
        while top_y > 0 and hpp_smoothed[top_y] > thresh:
            top_y -= 1
        valleys = [top_y]

        for i in range(len(peaks) - 1):
            p1, p2 = peaks[i], peaks[i + 1]
            valley_y = p1 + np.argmin(hpp_smoothed[p1:p2])
            valleys.append(valley_y)

        bottom_y = peaks[-1]
        while bottom_y < h - 1 and hpp_smoothed[bottom_y] > thresh:
            bottom_y += 1
        valleys.append(bottom_y)

        for idx in range(len(valleys) - 1):
            y1_r = valleys[idx]
            y2_r = valleys[idx + 1]
            if (y2_r - y1_r) >= min_line_height:
                raw_line_bounds.append((y1_r, y2_r))

    # Merge overlapping bounding boxes
    line_bounds = merge_overlapping_line_bounds(raw_line_bounds, overlap_threshold=0.30)

    line_crops = []
    debug_img = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    # Compute tight horizontal (X) bounds using Vertical Projection Profile (VPP)
    for idx, (y1_raw, y2_raw) in enumerate(line_bounds, start=1):
        y1 = max(0, y1_raw - line_padding)
        y2 = min(h, y2_raw + line_padding)

        line_binary = binary[y1:y2, :]
        vpp = np.sum(line_binary, axis=0)
        non_zero_x = np.where(vpp > 0)[0]

        if len(non_zero_x) > 0:
            x1 = max(0, non_zero_x[0] - line_padding)
            x2 = min(w, non_zero_x[-1] + line_padding)
        else:
            x1, x2 = 0, w

        line_crop = binary[y1:y2, x1:x2]

        line_crops.append({
            "line_idx": idx,
            "box": (x1, y1, x2, y2),
            "bbox": (x1, y1, x2 - x1, y2 - y1),
            "image": line_crop
        })

        # Draw tight green bounding boxes and line index labels
        cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        font_scale = max(0.40, w * 0.00045)
        cv2.putText(
            debug_img, 
            f"Line {idx}", 
            (x1 + 6, y1 + max(14, int((y2 - y1) * 0.4))),
            cv2.FONT_HERSHEY_SIMPLEX, 
            font_scale, 
            (255, 0, 0), 
            2
        )

    return line_crops, debug_img, binary