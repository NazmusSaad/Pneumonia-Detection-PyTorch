import streamlit as st
from PIL import Image
import torch
from util import classify, set_background
# from train import PneumoniaClassifier  # Import the class directly
from model import PneumoniaClassifier  # Now from model.py (not train.py)


# Create model instance
model = PneumoniaClassifier()
model.load_state_dict(torch.load('pneumonia_classifier.pth', map_location=torch.device('cpu')))
model.eval()

# UI Setup
# set_background('./bgs/bg5.png')
st.title('Pneumonia Classification')
st.header('Please upload an image of a chest X-ray image')

# Upload File
file = st.file_uploader('', type=['png', 'jpg', 'jpeg'])

# Load Labels
with open('labels.txt', 'r') as f:
    class_names = [line.strip().split(' ')[1] for line in f.readlines()]

# Process and Classify Image
if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, use_column_width=True)

    class_name, conf_score = classify(image, model, class_names)

    st.write(f"## {class_name}")
    st.write(f"### Score: {conf_score:.2f}%")
