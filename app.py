import streamlit as st
import numpy as np
import pickle
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas

# ------------------------------
# Load the trained model
# ------------------------------
model = pickle.load(open("gbr.pkl", "rb"))

# ------------------------------
# Streamlit Page Config
# ------------------------------
st.set_page_config(page_title="Digit & Alphabet Predictor", layout="centered")

st.title("✏️ Handwritten Digit & Alphabet Predictor")
st.write("Draw a **digit (0–9)** or **alphabet (A–Z)** below and click **Predict** to see the result!")

# ------------------------------
# Drawing Canvas
# ------------------------------
canvas_result = st_canvas(
    fill_color="white",  # background color
    stroke_width=12,
    stroke_color="black",
    background_color="white",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

# ------------------------------
# Prediction Section
# ------------------------------
if st.button("Predict"):
    if canvas_result.image_data is not None:
        # Convert canvas drawing to image
        img = Image.fromarray((255 - canvas_result.image_data[:, :, 0]).astype(np.uint8))
        img = ImageOps.grayscale(img)
        img = img.resize((28, 28))  # resize to model input size
        img_array = np.array(img).reshape(1, -1) / 255.0  # normalize and flatten
        
        # Make prediction
        try:
            pred = model.predict(img_array)[0]
            st.success(f"### 🧠 Predicted Character: **{pred}**")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
    else:
        st.warning("Please draw something first!")

# ------------------------------
# Optional: Image Upload
# ------------------------------
st.markdown("---")
st.write("Or upload an image o
