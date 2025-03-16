import base64
import streamlit as st
import torch
import numpy as np
from PIL import ImageOps, Image
import torchvision.transforms as transforms  # Add this line!

def set_background(image_file):
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{b64_encoded}");
            background-size: cover;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)

def classify(image, model, class_names):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]

    index = probabilities.argmax().item()
    class_name = class_names[index]
    confidence_score = probabilities[index].item() * 100

    return class_name, confidence_score
