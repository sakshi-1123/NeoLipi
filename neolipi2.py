# ============================================================
# NeoLipi - FastOCR With Preprocessing
# ============================================================

import os
import io
import time
import json
import tempfile
import base64
import uuid

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import requests

from PIL import Image

import streamlit as st

from dotenv import load_dotenv

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader


# ============================================================
# TRANSLATION IMPORT
# ============================================================
# IMPORTANT:
# This must match translation_utils.py
# ============================================================

from translation_utils import translate_sanskrit_to_kannada


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()


FASTOCR_API_KEY = os.getenv(
    "FASTOCR_API_KEY",
    ""
).strip()


FASTOCR_BASE_URL = "https://api.fastocr.org/v1"


# ============================================================
# STREAMLIT PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="NeoLipi - FastOCR",
    page_icon="🪨",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        background: linear-gradient(
            135deg,
            #f8fbff 0%,
            #eef4ff 100%
        );
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .neo-card {
        background: white;
        border-radius: 18px;
        padding: 22px;
        margin-bottom: 18px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.07);
    }

    .title {
        font-size: 38px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .subtitle {
        color: #667085;
        font-size: 17px;
    }

    .result-box {
        background: #ffffff;
        border-radius: 14px;
        padding: 20px;
        border: 1px solid #e5e7eb;
        font-size: 20px;
        line-height: 1.8;
    }

    .small-note {
        color: #667085;
        font-size: 14px;
    }

    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="title">🪨 NeoLipi</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Ancient Stone Inscription Recognition and Translation System'
    '</div>',
    unsafe_allow_html=True
)

st.write(
    "Stone Image → Grayscale → Bilateral Denoising → CLAHE → "
    "Background Normalization → FastOCR → Sanskrit/Devanagari "
    "→ Gemini → Kannada"
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Settings")


    if FASTOCR_API_KEY:

        st.success(
            "FastOCR API key detected"
        )

    else:

        st.error(
            "FASTOCR_API_KEY not found"
        )


    st.markdown("---")


    background_blur = st.slider(

        "Background normalization blur",

        min_value=31,

        max_value=151,

        value=71,

        step=2,

        help=(
            "Larger values estimate the stone background "
            "more smoothly."
        )
    )


    st.markdown("---")


    st.info(
        "FastOCR receives the final normalized image. "
        "No character segmentation or line segmentation "
        "is used."
    )


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def denoise_image(gray):

    """
    Bilateral denoising.

    Preserves inscription edges while reducing stone noise.
    """

    return cv2.bilateralFilter(

        gray,

        d=5,

        sigmaColor=30,

        sigmaSpace=30
    )


# ============================================================

def apply_clahe(gray):

    """
    Local contrast enhancement.
    """

    clahe = cv2.createCLAHE(

        clipLimit=1.5,

        tileGridSize=(8, 8)
    )

    return clahe.apply(gray)


# ============================================================

def normalize_background(
    gray,
    blur_size=71
):

    """
    Correct uneven stone/background illumination.

    normalized =
        gray / estimated_background * 128
    """

    if blur_size % 2 == 0:

        blur_size += 1


    background = cv2.GaussianBlur(

        gray,

        (blur_size, blur_size),

        0
    )


    background = np.maximum(

        background,

        1
    )


    normalized = cv2.divide(

        gray.astype(np.float32),

        background.astype(np.float32),

        scale=128
    )


    normalized = np.clip(

        normalized,

        0,

        255
    ).astype(np.uint8)


    return normalized


# ============================================================

def preprocess_image(
    image,
    blur_size=71
):

    """
    Complete NeoLipi preprocessing pipeline:

        Original
            ↓
        Grayscale
            ↓
        Bilateral denoising
            ↓
        CLAHE
            ↓
        Background normalization
            ↓
        FastOCR
    """

    gray = cv2.cvtColor(

        image,

        cv2.COLOR_BGR2GRAY
    )


    denoised = denoise_image(

        gray
    )


    enhanced = apply_clahe(

        denoised
    )


    normalized = normalize_background(

        enhanced,

        blur_size=blur_size
    )


    return {

        "gray": gray,

        "denoised": denoised,

        "enhanced": enhanced,

        "normalized": normalized
    }


# ============================================================
# IMAGE → PDF
# ============================================================

def image_to_pdf(
    image_path,
    pdf_path
):

    """
    Convert the preprocessed image into a PDF because
    FastOCR accepts the document/PDF upload in this workflow.
    """

    image = Image.open(

        image_path

    ).convert("RGB")


    width, height = image.size


    page_width, page_height = A4


    margin = 24


    scale = min(

        (page_width - 2 * margin) / width,

        (page_height - 2 * margin) / height
    )


    draw_width = width * scale

    draw_height = height * scale


    x = (
        page_width - draw_width
    ) / 2


    y = (
        page_height - draw_height
    ) / 2


    pdf = canvas.Canvas(

        pdf_path,

        pagesize=A4
    )


    pdf.drawImage(

        ImageReader(image),

        x,

        y,

        width=draw_width,

        height=draw_height,

        preserveAspectRatio=True,

        mask="auto"
    )


    pdf.showPage()

    pdf.save()


# ============================================================
# FASTOCR API HELPERS
# ============================================================

def fastocr_headers(
    json_content=False
):

    """
    FastOCR authentication.
    """

    headers = {

        "Authorization":
        f"Bearer {FASTOCR_API_KEY}"
    }


    if json_content:

        headers["Content-Type"] = (
            "application/json"
        )


    return headers


# ============================================================

def raise_fastocr_error(
    response,
    action
):

    """
    Convert an HTTP error into a useful Streamlit exception.
    """

    try:

        details = response.json()

    except Exception:

        details = response.text


    if isinstance(details, dict):

        detail_text = json.dumps(

            details,

            indent=2,

            ensure_ascii=False
        )

    else:

        detail_text = str(details)


    raise RuntimeError(

        f"FastOCR {action} failed "
        f"(HTTP {response.status_code}).\n\n"
        f"{detail_text}"
    )


# ============================================================
# CREATE FASTOCR DOCUMENT
# ============================================================

def create_fastocr_document(
    pdf_path
):

    """
    STEP 1:
    Create a FastOCR document job.
    """

    size_bytes = os.path.getsize(

        pdf_path
    )


    payload = {

        "filename":
        os.path.basename(pdf_path),

        "size_bytes":
        size_bytes
    }


    headers = fastocr_headers(

        json_content=True
    )


    headers["Idempotency-Key"] = str(

        uuid.uuid4()
    )


    url = (

        f"{FASTOCR_BASE_URL}/documents"
    )


    response = requests.post(

        url,

        headers=headers,

        json=payload,

        timeout=60
    )


    if not response.ok:

        raise_fastocr_error(

            response,

            "document creation"
        )


    data = response.json()


    document_id = data.get(
        "id"
    )


    upload_url = data.get(
        "upload_url"
    )


    if not document_id:

        raise RuntimeError(

            "FastOCR did not return "
            "a document ID:\n"

            + json.dumps(

                data,

                indent=2,

                ensure_ascii=False
            )
        )


    if not upload_url:

        raise RuntimeError(

            "FastOCR did not return "
            "a presigned upload URL:\n"

            + json.dumps(

                data,

                indent=2,

                ensure_ascii=False
            )
        )


    return data


# ============================================================
# UPLOAD FASTOCR PDF
# ============================================================

def upload_fastocr_pdf(
    upload_url,
    pdf_path
):

    """
    STEP 2:
    Upload PDF to FastOCR presigned S3 URL.
    """

    with open(

        pdf_path,

        "rb"

    ) as file:

        response = requests.put(

            upload_url,

            data=file,

            headers={

                "Content-Type":
                "application/pdf"
            },

            timeout=180,

            allow_redirects=False
        )


    if not response.ok:

        raise_fastocr_error(

            response,

            "PDF upload"
        )


    return True


# ============================================================
# START FASTOCR
# ============================================================

def start_fastocr_document(
    document_id
):

    """
    STEP 3:
    Start OCR processing.
    """

    url = (

        f"{FASTOCR_BASE_URL}/documents/"
        f"{document_id}/start"
    )


    response = requests.post(

        url,

        headers=fastocr_headers(),

        timeout=60
    )


    if not response.ok:

        raise_fastocr_error(

            response,

            "OCR start"
        )


    return response.json()


# ============================================================
# GET FASTOCR DOCUMENT
# ============================================================

def get_fastocr_document(
    document_id
):

    """
    STEP 4:
    Get document status.
    """

    url = (

        f"{FASTOCR_BASE_URL}/documents/"
        f"{document_id}"
    )


    response = requests.get(

        url,

        headers=fastocr_headers(),

        timeout=60
    )


    if not response.ok:

        raise_fastocr_error(

            response,

            "status request"
        )


    return response.json()


# ============================================================
# WAIT FOR FASTOCR
# ============================================================

def wait_for_fastocr(
    document_id,
    max_wait=300,
    interval=3
):

    """
    Poll FastOCR until the document reaches
    a terminal state.
    """

    start_time = time.time()


    while True:

        if (
            time.time() - start_time
            > max_wait
        ):

            raise TimeoutError(

                "FastOCR processing timed out."
            )


        data = get_fastocr_document(

            document_id
        )


        status = str(

            data.get(

                "status",

                ""
            )

        ).lower()


        if status in {

            "completed",

            "partial"
        }:

            return data


        if status == "failed":

            error_info = data.get(

                "error",

                data
            )


            raise RuntimeError(

                "FastOCR document processing failed:\n"

                + json.dumps(

                    error_info,

                    indent=2,

                    ensure_ascii=False
                )
            )


        time.sleep(

            interval
        )


# ============================================================
# DOWNLOAD FASTOCR TEXT
# ============================================================

def download_fastocr_text(
    document_id
):

    """
    STEP 5:
    Get extracted text.
    """

    url = (

        f"{FASTOCR_BASE_URL}/documents/"
        f"{document_id}/output"
    )


    response = requests.get(

        url,

        headers=fastocr_headers(),

        params={

            "format": "text"
        },

        timeout=60
    )


    if not response.ok:

        raise_fastocr_error(

            response,

            "text output request"
        )


    output_data = response.json()


    download_url = output_data.get(

        "url"
    )


    if not download_url:

        raise RuntimeError(

            "FastOCR did not return "
            "a text download URL:\n"

            + json.dumps(

                output_data,

                indent=2,

                ensure_ascii=False
            )
        )


    text_response = requests.get(

        download_url,

        timeout=120
    )


    if not text_response.ok:

        raise_fastocr_error(

            text_response,

            "text download"
        )


    return (

        text_response.text.strip(),

        output_data
    )


# ============================================================
# COMPLETE FASTOCR WORKFLOW
# ============================================================

def run_fastocr(
    pdf_path
):

    """
    Complete FastOCR Document OCR workflow:

        1. Create document
        2. Upload PDF
        3. Start OCR
        4. Poll status
        5. Download text
    """

    if not FASTOCR_API_KEY:

        raise RuntimeError(

            "FASTOCR_API_KEY is not configured."
        )


    with st.status(

        "Running FastOCR...",

        expanded=True

    ) as status:


        # ----------------------------------------------------
        # STEP 1
        # ----------------------------------------------------

        st.write(

            "1/5 Creating FastOCR document..."
        )


        document = create_fastocr_document(

            pdf_path
        )


        document_id = document["id"]

        upload_url = document["upload_url"]


        st.write(

            f"Document created: `{document_id}`"
        )


        # ----------------------------------------------------
        # STEP 2
        # ----------------------------------------------------

        st.write(

            "2/5 Uploading preprocessed PDF..."
        )


        upload_fastocr_pdf(

            upload_url,

            pdf_path
        )


        st.write(

            "PDF uploaded successfully."
        )


        # ----------------------------------------------------
        # STEP 3
        # ----------------------------------------------------

        st.write(

            "3/5 Starting OCR..."
        )


        start_fastocr_document(

            document_id
        )


        st.write(

            "OCR processing started."
        )


        # ----------------------------------------------------
        # STEP 4
        # ----------------------------------------------------

        st.write(

            "4/5 Waiting for FastOCR..."
        )


        final_data = wait_for_fastocr(

            document_id
        )


        final_status = str(

            final_data.get(

                "status",

                ""
            )

        ).lower()


        st.write(

            f"FastOCR status: `{final_status}`"
        )


        # ----------------------------------------------------
        # STEP 5
        # ----------------------------------------------------

        st.write(

            "5/5 Downloading extracted text..."
        )


        text, output_data = (

            download_fastocr_text(

                document_id
            )
        )


        status.update(

            label="FastOCR completed",

            state="complete"
        )


    return text, {

        "document":
        final_data,

        "output":
        output_data
    }


# ============================================================
# DOWNLOAD HELPERS
# ============================================================

def save_text_file(text):

    return text.encode(

        "utf-8"
    )


# ============================================================
# SESSION STATE
# ============================================================

if "ocr_text" not in st.session_state:

    st.session_state["ocr_text"] = ""


if "translation" not in st.session_state:

    st.session_state["translation"] = ""


if "raw_fastocr" not in st.session_state:

    st.session_state["raw_fastocr"] = None


# ============================================================
# 1. IMAGE UPLOAD
# ============================================================

st.header(

    "1. Upload Stone Inscription"
)


uploaded_file = st.file_uploader(

    "Upload an inscription image",

    type=[

        "png",

        "jpg",

        "jpeg",

        "bmp",

        "tif",

        "tiff"
    ],

    help=(
        "Upload a clear photograph or scan "
        "of the stone inscription."
    )
)


if uploaded_file is None:

    st.info(

        "Upload a stone inscription image to begin."
    )

    st.stop()


# ============================================================
# READ IMAGE
# ============================================================

file_bytes = np.asarray(

    bytearray(

        uploaded_file.getvalue()
    ),

    dtype=np.uint8
)


image_bgr = cv2.imdecode(

    file_bytes,

    cv2.IMREAD_COLOR
)


if image_bgr is None:

    st.error(

        "Unable to read the uploaded image."
    )

    st.stop()


# ============================================================
# DISPLAY ORIGINAL
# ============================================================

st.subheader(

    "Original Stone Image"
)


original_rgb = cv2.cvtColor(

    image_bgr,

    cv2.COLOR_BGR2RGB
)


st.image(

    original_rgb,

    caption="Original uploaded image",

    width="stretch"
)


# ============================================================
# 2. PREPROCESSING
# ============================================================

st.header(

    "2. Image Preprocessing"
)


with st.spinner(

    "Applying NeoLipi preprocessing..."
):

    processed = preprocess_image(

        image_bgr,

        blur_size=background_blur
    )


# ============================================================
# GRAYSCALE
# ============================================================

st.subheader(

    "2.1 Grayscale"
)


st.image(

    processed["gray"],

    caption="Grayscale",

    width="stretch"
)


# ============================================================
# BILATERAL
# ============================================================

st.subheader(

    "2.2 Bilateral Denoising"
)


st.image(

    processed["denoised"],

    caption="Bilateral denoising",

    width="stretch"
)


# ============================================================
# CLAHE
# ============================================================

st.subheader(

    "2.3 CLAHE Enhancement"
)


st.image(

    processed["enhanced"],

    caption="CLAHE enhanced",

    width="stretch"
)


# ============================================================
# NORMALIZATION
# ============================================================

st.subheader(

    "2.4 Background Normalization"
)


st.image(

    processed["normalized"],

    caption="Final normalized image sent to FastOCR",

    width="stretch"
)


# ============================================================
# SAVE PREPROCESSED IMAGE
# ============================================================

temp_dir = tempfile.mkdtemp(

    prefix="neolipi_"
)


preprocessed_path = os.path.join(

    temp_dir,

    "neolipi_preprocessed.png"
)


cv2.imwrite(

    preprocessed_path,

    processed["normalized"]
)


with open(

    preprocessed_path,

    "rb"

) as f:

    preprocessed_bytes = f.read()


st.download_button(

    "⬇️ Download Preprocessed Image",

    data=preprocessed_bytes,

    file_name="neolipi_preprocessed.png",

    mime="image/png"
)


# ============================================================
# CONVERT IMAGE TO PDF
# ============================================================

pdf_path = os.path.join(

    temp_dir,

    "neolipi_preprocessed.pdf"
)


try:

    image_to_pdf(

        preprocessed_path,

        pdf_path
    )

except Exception as e:

    st.error(

        f"Could not create PDF: {e}"
    )

    st.stop()


# ============================================================
# 3. FASTOCR
# ============================================================

st.header(

    "3. FastOCR"
)


st.write(

    "The complete normalized inscription image is sent "
    "to FastOCR. No character segmentation and no line "
    "segmentation are performed."
)


if not FASTOCR_API_KEY:

    st.error(

        "FASTOCR_API_KEY is missing."
    )


    st.info(

        "Create a .env file in the project folder with:\n\n"
        "FASTOCR_API_KEY=fok_live_YOUR_KEY"
    )

else:

    if st.button(

        "🔍 Extract Sanskrit Text with FastOCR",

        type="primary",

        use_container_width=True
    ):

        try:

            with st.spinner(

                "FastOCR is recognizing the inscription..."
            ):

                ocr_text, raw_result = run_fastocr(

                    pdf_path
                )


            st.session_state[

                "ocr_text"

            ] = ocr_text


            st.session_state[

                "raw_fastocr"

            ] = raw_result


            # Reset old translation

            st.session_state[

                "translation"

            ] = ""


        except requests.RequestException as e:

            st.error(

                f"FastOCR network/API request failed: {e}"
            )


        except Exception as e:

            st.error(

                f"FastOCR error: {e}"
            )


# ============================================================
# 4. OCR RESULT
# ============================================================

if st.session_state[

    "ocr_text"

].strip():


    st.header(

        "4. Sanskrit / Devanagari OCR Result"
    )


    ocr_text = st.session_state[

        "ocr_text"
    ]


    st.text_area(

        "Recognized text",

        value=ocr_text,

        height=300
    )


    st.download_button(

        "⬇️ Download Sanskrit OCR Text",

        data=save_text_file(

            ocr_text
        ),

        file_name="neolipi_sanskrit_ocr.txt",

        mime="text/plain"
    )


    # ========================================================
    # RAW FASTOCR RESPONSE
    # ========================================================

    with st.expander(

        "Show raw FastOCR response"
    ):

        st.json(

            st.session_state[

                "raw_fastocr"
            ]
        )


    # ========================================================
    # 5. GEMINI TRANSLATION
    # ========================================================

    st.header(

        "5. Sanskrit → Kannada Translation"
    )


    st.write(

        "Translation uses Google Gemini for "
        "Sanskrit → Kannada translation."
    )


    if st.button(

        "🌐 Translate to Kannada",

        use_container_width=True
    ):

        try:

            with st.spinner(

                "Gemini is translating Sanskrit to Kannada..."
            ):

                translation = (

                    translate_sanskrit_to_kannada(

                        ocr_text
                    )
                )


            st.session_state[

                "translation"

            ] = translation


        except Exception as e:

            st.error(

                f"Translation error: {e}"
            )


    # ========================================================
    # DISPLAY TRANSLATION
    # ========================================================

    if st.session_state[

        "translation"

    ].strip():


        st.subheader(

            "Kannada Translation"
        )


        st.text_area(

            "Translated text",

            value=st.session_state[

                "translation"
            ],

            height=300
        )


        st.download_button(

            "⬇️ Download Kannada Translation",

            data=save_text_file(

                st.session_state[

                    "translation"
                ]
            ),

            file_name="neolipi_kannada_translation.txt",

            mime="text/plain"
        )


# ============================================================
# INFORMATION
# ============================================================

with st.expander(

    "ℹ️ About the NeoLipi pipeline"
):

    st.markdown(
        """
        ### NeoLipi OCR Pipeline

        **Input**

        Ancient stone inscription image

        **Preprocessing**

        1. Grayscale conversion
        2. Bilateral denoising
        3. CLAHE contrast enhancement
        4. Background normalization

        **OCR**

        The complete preprocessed image is sent to FastOCR.

        **Important**

        - No character segmentation
        - No line segmentation
        - No individual-character classification
        - FastOCR receives the normalized image

        **Translation**

        Sanskrit/Devanagari OCR text is passed to the
        separate `translation_utils.py` module, which uses
        Google Gemini for Sanskrit → Kannada translation.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")


st.caption(

    "NeoLipi | Ancient Inscription Recognition | "
    "Bilateral Denoising → CLAHE → "
    "Background Normalization → FastOCR → "
    "Sanskrit → Gemini → Kannada"
)