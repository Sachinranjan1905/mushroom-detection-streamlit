import streamlit as st
import os
import json
import torch
import base64
from PIL import Image
from ultralytics import YOLO

# -------------------- CONFIG --------------------
st.set_page_config(page_title="🍄 Mushroom AI", layout="wide")

MODEL_PATH = "model/mushroom_model.pt"
INFO_PATH = "mushroom_info.json"

# -------------------- BACKGROUND IMAGE LOAD --------------------
def get_base64_image(image_path):
    with open(image_path, "rb") as img:
        return base64.b64encode(img.read()).decode()

bg_img = get_base64_image("static/images/bg.png")

# -------------------- CSS --------------------
st.markdown(f"""
<style>

/* 🌄 Background image + overlay */
.stApp {{
    background-image: 
        linear-gradient(rgba(255,255,255,0.85), rgba(255,255,255,0.85)),
        url("data:image/png;base64,{bg_img}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}}

/* 📦 Main container */
.block-container {{
    background: white;
    padding: 2rem;
    border-radius: 20px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    margin-top: 20px;
}}

/* 📝 GLOBAL TEXT FIX */
* {{
    color: #212121 !important;
    font-family: 'Segoe UI', sans-serif;
}}

/* 🏷️ Headings */
h1 {{
    color: #1b5e20 !important;
    font-weight: 700;
    text-align: center;
}}

h2, h3 {{
    color: #2e7d32 !important;
}}

/* 🎯 Buttons */
.stButton>button {{
    background: linear-gradient(45deg, #43a047, #66bb6a);
    color: white !important;
    border-radius: 12px;
    height: 45px;
    border: none;
}}

.stButton>button:hover {{
    transform: scale(1.03);
    transition: 0.2s;
}}

/* 📤 Upload main box */
.stFileUploader {{
    border: 2px dashed #66bb6a;
    border-radius: 15px;
    padding: 20px;
    background: #ffffff !important;
}}

/* 🚨 FULL UPLOADER FIX */
[data-testid="stFileUploader"] {{
    background: #ffffff !important;
}}

[data-testid="stFileUploaderDropzone"] {{
    background-color: #ffffff !important;
    border: 2px dashed #4caf50 !important;
    border-radius: 12px;
}}

/* All text inside uploader */
[data-testid="stFileUploader"] * {{
    color: #000000 !important;
    font-weight: 500;
}}

/* Drag text */
[data-testid="stFileUploaderDropzone"] div,
[data-testid="stFileUploaderDropzone"] span,
[data-testid="stFileUploaderDropzone"] p {{
    color: #000000 !important;
}}

/* File name */
[data-testid="stFileUploaderFileName"] {{
    color: #000000 !important;
}}

/* Browse button */
[data-testid="stFileUploader"] button {{
    color: #000000 !important;
    background-color: #e8f5e9 !important;
    border-radius: 8px;
}}

/* Icon color */
[data-testid="stFileUploader"] svg {{
    fill: #2e7d32 !important;
}}

/* 📂 Sidebar FIX */
[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #1b5e20, #2e7d32) !important;
    width: 260px !important;
    border-right: 2px solid #a5d6a7;
    box-shadow: 4px 0 10px rgba(0,0,0,0.2);
}}

/* Sidebar text ONLY */
[data-testid="stSidebar"] * {{
    color: #ffffff !important;
    font-size: 15px;
}}

/* Sidebar inputs */
[data-testid="stSidebar"] select {{
    color: black !important;
    background: #e8f5e9 !important;
    border-radius: 6px;
}}

/* Divider */
hr {{
    border: none;
    height: 2px;
    background: #c8e6c9;
}}

</style>
""", unsafe_allow_html=True)
# -------------------- SIDEBAR --------------------
st.sidebar.title("🍄 Mushroom AI")

menu = st.sidebar.radio("Navigation", [
    "Home",
    "Model Info",
    "Mushroom Types"
])

lang = st.sidebar.selectbox("🌐 Language", ["English", "Hindi"])

# -------------------- LOAD MODEL --------------------
@st.cache_resource
def load_model():
    try:
        if not os.path.exists(MODEL_PATH):
            return None
        model = YOLO(MODEL_PATH)
        model.to("cpu")
        return model
    except:
        return None

model = load_model()

# -------------------- LOAD DATA --------------------
@st.cache_data
def load_data():
    if os.path.exists(INFO_PATH):
        with open(INFO_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return {m["label"]: m for m in data}
            return data
    return {}

mushroom_data = load_data()

# -------------------- HOME --------------------
if menu == "Home":

    st.markdown("<h1>🍄🌱 Smart Mushroom Detection System</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;'>AI for Agriculture • Identify • Analyze • Learn</p>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📤 Upload Mushroom Image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file)

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="📸 Uploaded Image", use_container_width=True)

        with col2:
            if model is None:
                st.error("❌ Model not loaded")
            else:
                with st.spinner("🔍 Detecting..."):
                    try:
                        results = model.predict(image)
                        r = results[0]

                        label = None
                        names = model.names

                        if hasattr(r, "probs") and r.probs is not None:
                            label = names[int(r.probs.top1)]
                        elif hasattr(r, "boxes") and r.boxes is not None and len(r.boxes) > 0:
                            confs = r.boxes.conf
                            idx = int(torch.argmax(confs).item())
                            cls_id = int(r.boxes.cls[idx].item())
                            label = names[cls_id]

                        if label:
                            label = label.lower().replace(" ", "_")
                            mushroom = mushroom_data.get(label, {})

                            if lang == "Hindi":
                                st.success(f"✅ पहचान: {mushroom.get('name_hi', label)}")
                                st.write(mushroom.get("description_hi", ""))
                            else:
                                st.success(f"✅ Detected: {mushroom.get('name_en', label)}")
                                st.write(mushroom.get("description_en", ""))

                            st.markdown("### 🌱 Cultivation")
                            st.write(mushroom.get("cultivation_en", ""))

                            st.markdown("### 🥗 Nutrients")
                            for n in mushroom.get("nutrients", []):
                                st.write(f"✔️ {n}")

                            st.markdown("### 💰 Price")
                            for month, price in mushroom.get("price", {}).items():
                                st.write(f"{month}: ₹{price}/kg")

                        else:
                            st.warning("⚠️ No mushroom detected")

                    except Exception as e:
                        st.error(f"❌ Prediction error: {e}")

# -------------------- MODEL INFO --------------------
elif menu == "Model Info":

    st.title("🤖 Model Information")

    st.markdown("""
    ### 🔍 About the Model
    This project uses a YOLO-based deep learning model for detecting and classifying mushrooms.

    ### ⚙️ Details
    - Model: YOLOv12s  
    - Framework: PyTorch  
    - Type: Object Detection  
    - Application: Agriculture & Food Safety  

    ### 🎯 Purpose
    Helps farmers and consumers identify edible mushrooms using AI.
    """)

# -------------------- MUSHROOM TYPES --------------------
elif menu == "Mushroom Types":

    st.title("🍄 Mushroom Types")

    for key, mushroom in mushroom_data.items():

        if lang == "Hindi":
            st.markdown(f"### 🍄 {mushroom.get('name_hi', key)}")
            st.write(mushroom.get("description_hi", ""))
        else:
            st.markdown(f"### 🍄 {mushroom.get('name_en', key)}")
            st.write(mushroom.get("description_en", ""))

        st.markdown("---")