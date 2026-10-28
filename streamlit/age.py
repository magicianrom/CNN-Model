import streamlit as st
import numpy as np
from PIL import Image
from keras.models import load_model

# Load model
network = load_model("streamlit/age_model.keras")

st.title("Age Prediction")
st.write("Upload a facial image to predict the person's age.")

# Upload image
uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image")

    # Prepare image
    image = image.resize((120, 120))
    image_array = np.array(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    # Predict age automatically
    prediction = network.predict(image_array, verbose=0)[0][0]

    st.subheader("Prediction")
    st.write(f"Predicted age: {prediction:.1f} years")
