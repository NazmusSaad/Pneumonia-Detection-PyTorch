# **Pneumonia Detection Web App (PyTorch)**  

This Pneumonia Detection Web App utilizes a **custom CNN model built with PyTorch** to classify chest X-rays as **"Pneumonia" or "Normal."** The model achieves **85% accuracy** on a test dataset and is deployed using **Streamlit** for real-time image classification.  

## 🚀 **Key Features**  

- **PyTorch-Based CNN Model**: A convolutional neural network trained from scratch in PyTorch, achieving **85% test accuracy** in pneumonia detection.  
- **Streamlit Deployment**: The web app provides an intuitive interface for real-time image classification.  
- **Efficient Data Handling**: Uses PyTorch’s `DataLoader` for batch processing and optimized training.  
- **Real-Time Image Processing**: Integrated with **PIL (Pillow)** for image uploads and preprocessing.  
- **Separation of Training & Deployment**: The project follows best practices by separating training (`train.py`), model definition (`model.py`), and inference (`main.py`).  

---

## 🛠 **Technology Stack**  

- **Machine Learning**: PyTorch  
- **Web Framework**: Streamlit  
- **Image Processing**: PIL (Pillow)  
- **Language**: Python  

---

## 📂 **Project Structure**  

```
PneumoniaAIWebApp/
│── data/                 # Dataset (train/test folders)
│── model.py              # CNN architecture (PyTorch)
│── train.py              # Model training script
│── evaluate.py           # Model evaluation script
│── main.py               # Streamlit web app
│── util.py               # Helper functions (image preprocessing)
│── pneumonia_classifier.pth  # Trained PyTorch model
│── labels.txt            # Class labels
│── requirements.txt      # Dependencies
│── README.md             # Project documentation
```

---

## ⚡ **Installation & Usage**  

### 1️⃣ **Clone the Repository**  
```bash
git clone https://github.com/YOUR_USERNAME/Pneumonia-Detection-PyTorch.git
cd Pneumonia-Detection-PyTorch
```

### 2️⃣ **Install Dependencies**  
```bash
pip install -r requirements.txt
```

### 3️⃣ **Run the Web App**  
```bash
streamlit run main.py
```

### 4️⃣ **Train the Model (Optional)**
```bash
python train.py
```

### 5️⃣ **Evaluate Model Accuracy**
```bash
python evaluate.py
```

---

## 🖼 **Example Output**  
When an X-ray is uploaded, the app will display:  
✔️ The uploaded image  
✔️ The **predicted class (Normal/Pneumonia)**  
✔️ A **confidence score (%)**  
![Pneumonia Detection Example](https://raw.githubusercontent.com/NazmusSaad/Pneumonia-Detection-PyTorch/main/Screenshot%202025-03-16%20135059.png)



---

## 📌 **Why PyTorch?**
This project was originally implemented in TensorFlow/Keras but was re-implemented in PyTorch to:  
✅ Gain hands-on experience with **manual training loops and `DataLoader`**.  
✅ Compare PyTorch’s flexibility with **Keras’ high-level API**.  
✅ Follow industry trends, as **PyTorch is widely used in research & production**.  

---

## 🤖 **Next Steps & Future Work**  
🔹 Implement **transfer learning** with a pretrained ResNet model.  
🔹 Add **data augmentation** for improved generalization.  
🔹 Improve web app UI with **interactive confidence visualization**.  

---

### **📌 Want to Try It?**  
💻 **[Check out the repository here!](https://github.com/YOUR_USERNAME/Pneumonia-Detection-PyTorch)**  

🚀 If you like this project, **give it a ⭐ on GitHub!**  

