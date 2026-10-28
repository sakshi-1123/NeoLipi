NeoLipi — Multi-Script Ancient Inscription Recognition & Translation System

NeoLipi is an end-to-end computer vision and deep learning system built to analyze, segment, classify, recognize, and translate ancient stone inscriptions across three prominent historical scripts: Brahmi, Tamil, and Devanagari (Sanskrit).

Featuring an interactive Streamlit Web UI, NeoLipi supports both automatic OCR processing pipelines and interactive manual bounding-box annotation tools.

🌟 Key Features

Automatic Script Classification: EfficientNet-based classifier automatically detects whether an uploaded image contains Brahmi, Tamil, or Devanagari inscriptions, with confidence scoring and rejection thresholds for non-inscription images.

Brahmi OCR & Unicode Mapping: Full binarization, line detection, character cropping, and custom CNN classification mapped directly to Brahmi Unicode codepoints and Kannada character mappings via CSV dictionary tables.

Native Tamil OCR Pipeline: Morphological preprocessing (Bilateral filtering, Blackhat extraction, Otsu thresholding), HPP line strip isolation, bounding box character extraction, CNN recognition, dictionary-based post-correction, and direct Tamil-to-Kannada translation.

Devanagari / Sanskrit FastOCR: Preprocessing with normalized PDF conversions feeding cloud-assisted OCR and Sanskrit-to-Kannada translation capabilities.

Interactive Manual Segmentation Editor: Embedded bounding-box annotation component allowing users to draw and refine character crop regions per line strip for precision OCR re-computation.

Validation Guardrails: Confidence thresholding (<60% confidence triggers a "Classification Failed" alert) prevents invalid or non-script images from executing downstream pipelines.

🏗️ Project Architecture

├── app_final_6.py                 # Main Streamlit Application & Pipeline Controller
├── Model / models /               # Script classifiers & CNN model checkpoints
│   ├── script_classifier.keras    # Unified 3-Script Classifier (Brahmi, Tamil, Devanagari)
│   ├── cnn_brahmi.keras           # All-Class Brahmi Character Recognizer
│   └── cnn_brahmi.txt             # Brahmi Class Indices List
├── Brahmi /                       # Brahmi Binarization & Line/Character Segmentation
├── Tamil /                        # Tamil Morphological Filters & Verification Models
├── Devanagari /                   # Devanagari Normalization & FastOCR Connectors
├── Mapping /                      # Unicode & Transliteration Lookup Tables
├── brahmi_mapping.csv             # Class label to Unicode/Kannada Glyph mapping
├── requirements.txt               # Dependencies
└── README.md                      # Documentation


🚀 Getting Started

1. Prerequisites

Ensure you have Python 3.9+ installed on your system.

2. Clone the Repository

git clone https://github.com/your-username/neolipi-ocr.git
cd neolipi-ocr


3. Install Dependencies

pip install -r requirements.txt


Example core requirements:

streamlit
opencv-python
numpy
pandas
pillow
tensorflow
deep-translator
python-dotenv
streamlit-drawable-canvas


4. Configure Environment Variables (Optional)

For Devanagari FastOCR or Gemini translation capabilities, create an env or .env file inside the Devanagari/ directory:

FASTOCR_API_KEY=your_fastocr_api_key
GEMINI_API_KEY=your_gemini_api_key


5. Run the Application

streamlit run app_final_6.py


Open your browser and navigate to http://localhost:8501.

📸 User Workflow

Upload Inscription Image: Select a stone inscription image (.png, .jpg, .jpeg, .webp, .tif).

Script Classification: NeoLipi inspects the image and identifies whether it belongs to Brahmi, Tamil, or Devanagari.

Analyze: Click the Analyze button to trigger the script-specific binarization, line extraction, and character recognition pipeline.

Interactive Correction (Optional): Switch to Manual Segmentation Mode in the sidebar to manually draw or adjust bounding boxes over lines to re-run character recognition in real-time.

View Results: Inspect extracted characters, confidence scores, Digital Unicode text, Transliterated text, and modern Kannada translations.
