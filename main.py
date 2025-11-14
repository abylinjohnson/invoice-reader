"""
Streamlit + Tesseract Invoice Readergsk_Yu1HfxdSbgRrFY8R6HuVWGdyb3FYJ5dStC0eCoJhKTlCM8T7cFrA
How to use:
1. Install system Tesseract engine:
   - Ubuntu: sudo apt-get install tesseract-ocr
   - macOS (Homebrew): brew install tesseract
   - Windows: install from https://github.com/tesseract-ocr/tesseract/releases and add to PATH

2. (Optional for PDF support) Install poppler:
   - Ubuntu: sudo apt-get install poppler-utils
   - macOS (Homebrew): brew install poppler
   - Windows: download poppler binaries and add to PATH

3. Python deps:
   pip install -r requirements.txt
   where requirements.txt contains:
       streamlit
       pytesseract
       pillow
       opencv-python-headless
       numpy
       pandas
       pdf2image  # optional, for PDFs

Run:
    streamlit run streamlit_tesseract_invoice_reader.py

This app accepts image files (jpg, png, tif) and PDFs (first page) and returns OCR text and a set of parsed fields (invoice number, date, total/amount).
"""

import streamlit as st
import pytesseract
from PIL import Image
import cv2
import numpy as np
import io
import re
import pandas as pd

# Optional: pdf -> image
try:
    from pdf2image import convert_from_bytes
    PDF2IMAGE_AVAILABLE = True
except Exception:
    PDF2IMAGE_AVAILABLE = False

st.set_page_config(page_title="Invoice Reader", layout="wide")

# ---------- Utility functions ----------

def load_image_from_bytes(file_bytes):
    """Load image from raw bytes into OpenCV BGR image."""
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def preprocess_for_ocr(bgr_image):
    """Preprocess image for better OCR:
    - convert to gray
    - denoise
    - adaptive thresholding
    - dilation/erosion optional
    Returns PIL.Image ready for pytesseract
    """
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

    # Resize if too small
    h, w = gray.shape
    if max(h, w) < 1000:
        scale = 1000 / max(h, w)
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

    # Denoise
    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    # Adaptive threshold
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 31, 11)

    # Morphological open to remove small speckles
    kernel = np.ones((1, 1), np.uint8)
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Convert to PIL for pytesseract
    pil_img = Image.fromarray(opened)
    return pil_img

from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def parse_invoice_text(ocr_text):
    """
    Sends OCR text to OpenAI using streaming and returns a full invoice description.
    """
    try:
        load_dotenv()
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Build prompt
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert invoice summarizer. "
                    "Read the provided OCR text and generate a clean, accurate description of the invoice. "
                    "Do NOT hallucinate missing info. Only use what is visible in the OCR text."
                )
            },
            {
                "role": "user",
                "content": f"OCR TEXT:\n\n{ocr_text}\n\nGenerate a description:"
            }
        ]

        # Streaming response
        stream = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            max_tokens=1024,
            stream=True,
            temperature=0.1,
        )

        full_output = ""

        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                full_output += delta.content

        return {
            "description": full_output.strip()
        }

    except Exception as e:
        return {"error": str(e)}

# ---------- Streamlit UI ----------

st.title("Invoice Data Extractor")
st.write("Upload invoice images or PDFs and extract text")

uploaded = st.file_uploader("Upload image or PDF", type=["png", "jpg", "jpeg", "tif", "tiff", "pdf"], accept_multiple_files=False)

if uploaded is None:
    st.info("Upload a file to get started. For PDF support, install pdf2image and poppler.")
    st.stop()

file_bytes = uploaded.read()
filetype = uploaded.type

images = []
if uploaded.name.lower().endswith('.pdf'):
    if not PDF2IMAGE_AVAILABLE:
        st.error("PDF support requires pdf2image. Install it (pip install pdf2image) and system poppler.")
        st.stop()
    try:
        pil_pages = convert_from_bytes(file_bytes, dpi=300)
        images = []
        for p in pil_pages:
            images.append(cv2.cvtColor(np.array(p.convert('RGB')), cv2.COLOR_RGB2BGR))
    except Exception as e:
        st.error(f"Failed to convert PDF to image: {e}")
        st.stop()
else:
    try:
        img = load_image_from_bytes(file_bytes)
        images = [img]
    except Exception as e:
        st.error(f"Failed to load image: {e}")
        st.stop()

# If multiple pages, allow selection
if len(images) > 1:
    page = st.number_input("Page number", min_value=1, max_value=len(images), value=1)
    img_bgr = images[page-1]
else:
    img_bgr = images[0]

col1, col2 = st.columns([1,1])

with col1:
    st.subheader("Image preview")
    # show using PIL
    pil_vis = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    st.image(pil_vis, use_column_width=True)

with col2:
    st.subheader("OCR & parsed data")

    preprocess_checkbox = st.checkbox("Apply preprocessing for OCR (recommended)", value=True)
    oem_psm_options = {
        "Default": "",
        "Single block": "--psm 6",
        "Single line": "--psm 7",
        "Single word": "--psm 8",
    }
    psm_choice = st.selectbox("OCR mode (PSM)", list(oem_psm_options.keys()))
    config_extra = oem_psm_options[psm_choice]

    if preprocess_checkbox:
        pil_for_ocr = preprocess_for_ocr(img_bgr)
    else:
        pil_for_ocr = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

    try:
        ocr_result = pytesseract.image_to_string(pil_for_ocr, config=config_extra)
    except Exception as e:
        st.error(f"Tesseract OCR failed: {e}\nMake sure tesseract is installed and available in PATH.")
        st.stop()

    parsed = parse_invoice_text(ocr_result)

    st.markdown("**Raw OCR output**")
    st.text_area("OCR Text", value=ocr_result, height=240)

    st.markdown("**Description from LLM**")
    st.write(parsed)

st.write("---")
st.caption("This tool performs best on clean, high-resolution invoices. For production use consider training a layout-aware model (e.g., LayoutLM/Donut) or adding advanced layout detection + table parsing.")
