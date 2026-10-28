import cv2
import numpy as np
import os
import argparse

# ============================================================
# NeoLipi Brahmi Character Segmentation V6 + Reconstruction
# ============================================================
#
# PURPOSE
# -------
# Robust, stroke-preserving segmentation for Brahmi stone
# inscriptions with integrated character rebuilding/reconstruction
# for broken strokes prior to and following segmentation.
#
# ============================================================

OUTPUT_DIR = "segmentation_v6"

# ------------------------------------------------------------
# Noise / component limits
# ------------------------------------------------------------
MIN_COMPONENT_AREA = 8
MIN_COMPONENT_HEIGHT = 4
MIN_COMPONENT_WIDTH = 2

# ------------------------------------------------------------
# Analysis morphology
# ------------------------------------------------------------
ANALYSIS_CLOSE_X = 3
ANALYSIS_CLOSE_Y = 2

# ------------------------------------------------------------
# Row detection
# ------------------------------------------------------------
ROW_GAP_RATIO = 0.32
ROW_MIN_HEIGHT_RATIO = 0.012
ROW_MIN_COMPONENTS = 2

# ------------------------------------------------------------
# Character grouping
# ------------------------------------------------------------
TINY_WIDTH_RATIO = 0.38
TINY_AREA_RATIO = 0.18
FRAGMENT_GAP_RATIO = 0.42
MAX_MERGED_WIDTH_RATIO = 1.45

# ------------------------------------------------------------
# Wide-group splitting
# ------------------------------------------------------------
WIDE_RATIO = 2.15
VERY_WIDE_RATIO = 3.20
VALLEY_THRESHOLD = 0.32
MIN_PART_WIDTH_RATIO = 0.48
MIN_CUT_DISTANCE_RATIO = 0.42

# ------------------------------------------------------------
# Final crop
# ------------------------------------------------------------
CROP_PADDING_RATIO = 0.12

# ============================================================
# REBUILDING / RECONSTRUCTION ENGINE
# ============================================================

_NEIGHBOR_OFFSETS = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]

def zhang_suen_thinning(binary):
    """Skeletonize a binary image (foreground=255) using Zhang-Suen thinning."""
    img = (binary > 0).astype(np.uint8)
    changed = True

    while changed:
        changed = False
        for step in (0, 1):
            padded = np.pad(img, 1, mode="constant")
            P2 = padded[0:-2, 1:-1]
            P3 = padded[0:-2, 2:]
            P4 = padded[1:-1, 2:]
            P5 = padded[2:, 2:]
            P6 = padded[2:, 1:-1]
            P7 = padded[2:, 0:-2]
            P8 = padded[1:-1, 0:-2]
            P9 = padded[0:-2, 0:-2]

            B = (P2 + P3 + P4 + P5 + P6 + P7 + P8 + P9)
            neighbors = [P2, P3, P4, P5, P6, P7, P8, P9, P2]
            A = np.zeros_like(img)

            for k in range(8):
                A += ((neighbors[k] == 0) & (neighbors[k + 1] == 1)).astype(np.uint8)

            if step == 0:
                cond3 = (P2 * P4 * P6) == 0
                cond4 = (P4 * P6 * P8) == 0
            else:
                cond3 = (P2 * P4 * P8) == 0
                cond4 = (P2 * P6 * P8) == 0

            marker = (
                (img == 1)
                & (B >= 2) & (B <= 6)
                & (A == 1)
                & cond3
                & cond4
            )

            if marker.any():
                img[marker] = 0
                changed = True

    return (img * 255).astype(np.uint8)


def find_endpoints(skeleton):
    """Find stroke termination points in a skeleton image."""
    ys, xs = np.where(skeleton > 0)
    pts = set(zip(ys.tolist(), xs.tolist()))
    endpoints = []

    for (y, x) in pts:
        n = 0
        for dy, dx in _NEIGHBOR_OFFSETS:
            if (y + dy, x + dx) in pts:
                n += 1
        if n <= 1:
            endpoints.append((y, x))

    return endpoints


def label_components(skeleton):
    n, labels = cv2.connectedComponents(skeleton, connectivity=8)
    return n, labels


def estimate_stroke_width(binary):
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    vals = dist[binary > 0]
    if vals.size == 0:
        return 2
    width = 2.0 * np.median(vals)
    return max(1, int(round(width)))


def _join_broken_pieces(
    binary,
    threshold_px,
    min_component_pixels=3,
    max_merged_w=None,
    max_merged_h=None,
):
    """Core endpoint-joining algorithm to bridge broken strokes."""
    skeleton = zhang_suen_thinning(binary)
    n_labels, labels = label_components(skeleton)

    comp_sizes = np.bincount(labels.ravel())
    valid_labels = [
        i for i in range(1, n_labels)
        if comp_sizes[i] >= min_component_pixels
    ]
    valid_label_set = set(valid_labels)

    joins = []
    if len(valid_label_set) < 2:
        return binary, skeleton, joins

    bbox = {}
    for lbl in valid_labels:
        ys, xs = np.where(labels == lbl)
        bbox[lbl] = [int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())]

    parent = {lbl: lbl for lbl in valid_labels}

    def find(lbl):
        while parent[lbl] != lbl:
            parent[lbl] = parent[parent[lbl]]
            lbl = parent[lbl]
        return lbl

    def union_bbox(l1, l2):
        r1, r2 = find(l1), find(l2)
        b1, b2 = bbox[r1], bbox[r2]
        merged = [
            min(b1[0], b2[0]), max(b1[1], b2[1]),
            min(b1[2], b2[2]), max(b1[3], b2[3]),
        ]
        parent[r2] = r1
        bbox[r1] = merged
        return r1

    endpoints_before = find_endpoints(skeleton)
    ep_with_label = []
    for (y, x) in endpoints_before:
        lbl = labels[y, x]
        if lbl in valid_label_set:
            ep_with_label.append((y, x, lbl))

    stroke_width = estimate_stroke_width(binary)
    working = binary.copy()
    remaining = ep_with_label

    while True:
        candidates = []
        for i in range(len(remaining)):
            y1, x1, l1 = remaining[i]
            for j in range(i + 1, len(remaining)):
                y2, x2, l2 = remaining[j]

                r1, r2 = find(l1), find(l2)
                if r1 == r2:
                    continue

                d = np.hypot(y1 - y2, x1 - x2)
                if d <= threshold_px:
                    candidates.append((d, i, j))

        candidates.sort(key=lambda t: t[0])
        joined = False

        for d, i, j in candidates:
            y1, x1, l1 = remaining[i]
            y2, x2, l2 = remaining[j]

            r1, r2 = find(l1), find(l2)

            if max_merged_w is not None or max_merged_h is not None:
                b1, b2 = bbox[r1], bbox[r2]
                merged_h = max(b1[1], b2[1]) - min(b1[0], b2[0]) + 1
                merged_w = max(b1[3], b2[3]) - min(b1[2], b2[2]) + 1

                if max_merged_w is not None and merged_w > max_merged_w:
                    continue
                if max_merged_h is not None and merged_h > max_merged_h:
                    continue

            cv2.line(working, (x1, y1), (x2, y2), 255, thickness=stroke_width)
            joins.append(((x1, y1), (x2, y2)))

            union_bbox(l1, l2)

            new_remaining = [
                (yy, xx, ll) for k, (yy, xx, ll) in enumerate(remaining)
                if k not in (i, j)
            ]
            remaining = new_remaining
            joined = True
            break

        if not joined:
            break

    return working, skeleton, joins


def bridge_gaps_in_mask(
    mask,
    max_join_distance_px,
    min_component_pixels=3,
    max_merged_w=None,
    max_merged_h=None,
):
    """Bridge stroke gaps at line-level prior to character bounding box calculation."""
    if mask is None or mask.size == 0:
        return mask, []

    binary = (mask > 0).astype(np.uint8) * 255
    working, _, joins = _join_broken_pieces(
        binary,
        max_join_distance_px,
        min_component_pixels,
        max_merged_w=max_merged_w,
        max_merged_h=max_merged_h,
    )
    return working, joins


def rebuild_character(
    char_crop,
    max_join_distance_ratio=0.65,
    min_component_pixels=3,
    return_debug=False,
):
    """Rebuild a individual broken character crop."""
    if char_crop is None or char_crop.size == 0:
        return char_crop if not return_debug else (char_crop, {})

    binary = (char_crop > 0).astype(np.uint8) * 255
    h, w = binary.shape

    n_labels_check, labels_check = label_components(zhang_suen_thinning(binary))
    comp_sizes = np.bincount(labels_check.ravel())
    n_valid = sum(
        1 for i in range(1, n_labels_check)
        if comp_sizes[i] >= min_component_pixels
    )

    if n_valid < 2:
        rebuilt = binary
        joins = []
        skeleton = zhang_suen_thinning(binary)
    else:
        diagonal = float(np.hypot(h, w))
        threshold_px = max_join_distance_ratio * diagonal
        rebuilt, skeleton, joins = _join_broken_pieces(
            binary, threshold_px, min_component_pixels
        )

    if not return_debug:
        return rebuilt

    skeleton_after = zhang_suen_thinning(rebuilt)
    debug = {
        "skeleton_before": skeleton,
        "skeleton_after": skeleton_after,
        "endpoints_before": find_endpoints(skeleton),
        "joins": joins,
    }
    return rebuilt, debug

# ============================================================
# IO & HELPERS
# ============================================================

def ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "lines"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "characters"), exist_ok=True)


def save(relative_path, image):
    path = os.path.join(OUTPUT_DIR, relative_path)
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    cv2.imwrite(path, image)
    print("[SAVED]", path)


def load_binary(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Cannot read binary image:\n{path}")
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    if np.mean(binary > 0) > 0.60:
        binary = cv2.bitwise_not(binary)
    return binary


def overlap_1d(a1, a2, b1, b2):
    return max(0, min(a2, b2) - max(a1, b1))


def box_gap(a, b):
    ax, ay, aw, ah = a[:4]
    bx, by, bw, bh = b[:4]

    ax2, bx2 = ax + aw, bx + bw
    gx = bx - ax2 if ax2 < bx else (ax - bx2 if bx2 < ax else 0)

    ay2, by2 = ay + ah, by + bh
    gy = by - ay2 if ay2 < by else (ay - by2 if by2 < ay else 0)

    return gx, gy


def filter_components(components, min_aspect=0.1, max_aspect=4.0):
    if not components:
        return []

    widths = [c[2] for c in components]
    heights = [c[3] for c in components]
    areas = [c[4] for c in components]

    med_w = np.median(widths)
    med_h = np.median(heights)
    med_area = np.median(areas)

    filtered = []
    for comp in components:
        x, y, w, h, area = comp
        aspect_ratio = w / float(h)

        if area < 0.15 * med_area or w < 0.2 * med_w or h < 0.25 * med_h:
            continue
        if area > 6.0 * med_area or w > 4.0 * med_w or h > 4.0 * med_h:
            continue
        if aspect_ratio < min_aspect or aspect_ratio > max_aspect:
            continue

        filtered.append(comp)

    return filtered


def remove_tiny_components(binary):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    out = np.zeros_like(binary)

    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])

        if (
            area >= MIN_COMPONENT_AREA
            and w >= MIN_COMPONENT_WIDTH
            and h >= MIN_COMPONENT_HEIGHT
        ):
            out[labels == i] = 255

    return out


def remove_border_components(binary):
    h, w = binary.shape
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    out = binary.copy()

    for i in range(1, n):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        cw = int(stats[i, cv2.CC_STAT_WIDTH])
        ch = int(stats[i, cv2.CC_STAT_HEIGHT])

        touches = (x <= 0) or (x + cw >= w) or (y <= 0) or (y + ch >= h)
        if not touches:
            continue

        horizontal_strip = cw >= int(0.55 * w) and ch <= max(8, int(0.08 * h))
        vertical_strip = ch >= int(0.55 * h) and cw <= max(8, int(0.08 * w))
        huge = cw >= int(0.85 * w) or ch >= int(0.85 * h)

        if horizontal_strip or vertical_strip or huge:
            out[labels == i] = 0

    return out


# ============================================================
# ROW DETECTION
# ============================================================

def row_components(row):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(row, connectivity=8)
    boxes = []

    for i in range(1, n):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        area = int(stats[i, cv2.CC_STAT_AREA])

        if area < MIN_COMPONENT_AREA or w < MIN_COMPONENT_WIDTH or h < MIN_COMPONENT_HEIGHT:
            continue

        boxes.append((x, y, w, h, area))

    return boxes


def _cluster_vertical_intervals(boxes, image_height):
    if not boxes:
        return []

    heights = np.array([b[3] for b in boxes], dtype=np.float32)
    cutoff = max(4.0, float(np.percentile(heights, 35)))
    useful = [b for b in boxes if b[3] >= cutoff]

    if len(useful) < ROW_MIN_COMPONENTS:
        useful = boxes

    useful.sort(key=lambda b: (b[1], b[0]))
    med_h = max(5.0, float(np.median([b[3] for b in useful])))
    gap_tol = max(2, int(round(ROW_GAP_RATIO * med_h)))

    bands = []
    cur_y1, cur_y2 = useful[0][1], useful[0][1] + useful[0][3]

    for b in useful[1:]:
        y1, y2 = b[1], b[1] + b[3]
        if y1 <= cur_y2 + gap_tol:
            cur_y2 = max(cur_y2, y2)
        else:
            bands.append((cur_y1, cur_y2))
            cur_y1, cur_y2 = y1, y2

    bands.append((cur_y1, cur_y2))
    min_band_h = max(6, int(round(ROW_MIN_HEIGHT_RATIO * image_height)))
    bands = [b for b in bands if (b[1] - b[0]) >= min_band_h]

    return bands


def detect_lines(binary):
    if binary is None:
        return [], [], np.array([], dtype=np.float32)

    mask = remove_border_components(binary)
    mask = remove_tiny_components(mask)

    boxes = row_components(mask)
    rows = _cluster_vertical_intervals(boxes, mask.shape[0])

    if not rows:
        ys, xs = np.where(mask > 0)
        rows = [(int(ys.min()), int(ys.max()) + 1)] if len(ys) else [(0, mask.shape[0])]

    line_boxes = [(0, int(y1), int(mask.shape[1]), int(y2)) for y1, y2 in rows]
    projection = np.sum(mask > 0, axis=1).astype(np.float32)
    peaks = []

    return line_boxes, peaks, projection

# ============================================================
# SEGMENTATION WITH PRE-RECONSTRUCTION
# ============================================================

def estimate_normal_character_width(components):
    if not components:
        return 8.0
    widths = np.array([c[2] for c in components], dtype=np.float32)
    upper = np.percentile(widths, 75)
    normal = widths[widths <= upper]
    if len(normal) == 0:
        normal = widths
    return max(5.0, float(np.median(normal)))


def merge_fragments(components, normal_w):
    if not components:
        return []

    comps = [list(c) for c in sorted(components, key=lambda b: b[0])]
    areas = np.array([c[4] for c in comps], dtype=np.float32)
    normal_area = max(10.0, float(np.median(areas)))

    used = [False] * len(comps)
    order = sorted(range(len(comps)), key=lambda i: (comps[i][4], comps[i][2]), reverse=True)
    result = []

    for i in order:
        if used[i]:
            continue

        base = comps[i]
        used[i] = True
        changed = True

        while changed:
            changed = False
            bx, by, bw, bh, ba = base

            for j in order:
                if used[j]:
                    continue

                x, y, w, h, area = comps[j]
                tiny = (w <= TINY_WIDTH_RATIO * normal_w or area <= TINY_AREA_RATIO * normal_area)
                if not tiny:
                    continue

                gx, gy = box_gap(base, comps[j])
                max_gap = max(2, int(round(FRAGMENT_GAP_RATIO * normal_w)))
                overlap_y = overlap_1d(by, by + bh, y, y + h)
                overlap_ratio = overlap_y / float(max(1, min(bh, h)))

                if gx <= max_gap and (overlap_ratio >= 0.20 or gy <= max(2, int(0.18 * bh))):
                    new_x1, new_y1 = min(bx, x), min(by, y)
                    new_x2, new_y2 = max(bx + bw, x + w), max(by + bh, y + h)
                    new_w, new_h = new_x2 - new_x1, new_y2 - new_y1

                    if new_w <= MAX_MERGED_WIDTH_RATIO * normal_w:
                        base = [new_x1, new_y1, new_w, new_h, ba + area]
                        used[j] = True
                        changed = True
                        bx, by, bw, bh, ba = base

        result.append(tuple(base))

    result.sort(key=lambda b: b[0])
    return result


def segment_line(line_roi):
    """
    Extracts individual character bounding boxes within a single line image.
    Applies text reconstruction/rebuilding prior to component extraction.
    """
    if line_roi is None or line_roi.size == 0:
        return line_roi, []

    # RECONSTRUCTION BEFORE SEGMENTATION: Bridge gaps across broken character strokes
    reconstructed_roi, _ = bridge_gaps_in_mask(
        line_roi,
        max_join_distance_px=12,
        min_component_pixels=3,
        max_merged_w=int(line_roi.shape[0] * 1.5),
        max_merged_h=line_roi.shape[0]
    )

    # Analysis morphology on reconstructed line
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 2))
    analysis_roi = cv2.morphologyEx(reconstructed_roi, cv2.MORPH_CLOSE, kernel)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(analysis_roi, connectivity=8)

    raw_components = []
    for i in range(1, n):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        area = int(stats[i, cv2.CC_STAT_AREA])

        if area < MIN_COMPONENT_AREA or w < MIN_COMPONENT_WIDTH or h < MIN_COMPONENT_HEIGHT:
            continue

        raw_components.append((x, y, w, h, area))

    if not raw_components:
        return reconstructed_roi, []

    normal_w = estimate_normal_character_width(raw_components)
    components = merge_fragments(raw_components, normal_w)
    components.sort(key=lambda c: c[0])

    return reconstructed_roi, components


def extract_character(full_binary, line_box, component):
    """
    Extracts the cropped character, running final endpoint reconstruction
    on the crop itself to bridge any micro-gaps.
    """
    lx1, ly1, lx2, ly2 = line_box
    cx, cy, cw, ch = component[:4]

    gx1 = lx1 + cx
    gy1 = ly1 + cy
    gx2 = gx1 + cw
    gy2 = gy1 + ch

    pad = max(2, int(round(CROP_PADDING_RATIO * max(cw, ch))))
    H, W = full_binary.shape[:2]

    px1 = max(0, gx1 - pad)
    py1 = max(0, gy1 - pad)
    px2 = min(W, gx2 + pad)
    py2 = min(H, gy2 + pad)

    crop = full_binary[py1:py2, px1:px2].copy()
    
    # Final per-character crop reconstruction pass
    rebuilt_crop = rebuild_character(crop, max_join_distance_ratio=0.65)
    return rebuilt_crop


# ============================================================
# VISUALIZATION & PROCESS
# ============================================================

def draw_line_boxes(binary, line_boxes):
    canvas = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    for i, (x1, y1, x2, y2) in enumerate(line_boxes, start=1):
        cv2.rectangle(canvas, (0, y1), (binary.shape[1] - 1, y2 - 1), (255, 255, 0), 2)
        cv2.putText(canvas, f"Line {i}", (5, max(15, y1 + 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 0), 1, cv2.LINE_AA)
    return canvas


def draw_character_boxes(binary, boxes):
    canvas = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    for number, item in enumerate(boxes, start=1):
        x1, y1, x2, y2 = item[1:5] if len(item) == 5 else item[:4]
        cv2.rectangle(canvas, (x1, y1), (x2 - 1, y2 - 1), (0, 255, 0), 1)
        cv2.putText(canvas, str(number), (x1, max(12, y1 - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 255, 0), 1, cv2.LINE_AA)
    return canvas


def process(binary_path):
    ensure_dirs()
    original = load_binary(binary_path)
    h, w = original.shape

    save("01_BINARY_INPUT.png", original)
    line_boxes, _, _ = detect_lines(original)
    save("02_LINE_BOXES.png", draw_line_boxes(original, line_boxes))

    all_boxes = []
    total = 0

    for line_no, line_box in enumerate(line_boxes, start=1):
        lx1, ly1, lx2, ly2 = line_box
        line = original[ly1:ly2, lx1:lx2].copy()

        analysis, components = segment_line(line)
        save(f"lines/line_{line_no:02d}.png", line)
        save(f"lines/line_{line_no:02d}_reconstructed.png", analysis)

        local_boxes = []
        for char_no, component in enumerate(components, start=1):
            x, y, cw, ch, area = component
            crop = extract_character(original, line_box, component)

            gx1, gy1 = lx1 + x, ly1 + y
            gx2, gy2 = gx1 + cw, gy1 + ch

            all_boxes.append((line_no, gx1, gy1, gx2, gy2))
            local_boxes.append((line_no, x, y, x + cw, y + ch))

            save(
              f"characters/char_{total + 1:03d}.png",
              crop
            )
            total += 1

        save(
            f"lines/line_{line_no:02d}_character_boxes.png",
            draw_character_boxes(analysis, local_boxes),
        )

    all_boxes.sort(key=lambda b: (b[0], b[1]))
    save("03_CHARACTER_BOXES.png", draw_character_boxes(original, all_boxes))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NeoLipi Brahmi Character Segmentation V6 with Reconstruction")
    parser.add_argument("binary", help="Path to binary image")
    args = parser.parse_args()

    process(args.binary)