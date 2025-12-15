import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import cv2
from streamlit_drawable_canvas import st_canvas

# Page config
st.set_page_config(
    page_title="MNIST Digit Classifier",
    page_icon="🔢",
    layout="wide"
)

# Load model
@st.cache_resource
def load_mnist_model():
    return load_model('C:\\Users\\fares\\Downloads\\projet\\TP5\\expoted_Models\\best_mnist_model.h5')

model = load_mnist_model()

# Title
st.title("🔢 MNIST Digit Classifier")
st.markdown("Draw a digit or upload an image to classify!")

# Tabs
tab1, tab2 = st.tabs(["📝 Draw", "📤 Upload"])

# Tab 1: Drawing Canvas
with tab1:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Draw a digit (0-9)")
        
        # Create a canvas
        canvas_result = st_canvas(
            fill_color="black",
            stroke_width=20,
            stroke_color="white",
            background_color="black",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="canvas",
        )
        
        if st.button("🎯 Predict", key="predict_draw"):
            if canvas_result.image_data is not None:
                # Preprocess canvas data
                img = canvas_result.image_data[:, :, 0]  # Get one channel
                img = cv2.resize(img, (28, 28))
                img = img.astype('float32') / 255.0
                img = img.reshape(1, 28, 28, 1)
                
                # Predict
                predictions = model.predict(img, verbose=0)
                predicted_digit = np.argmax(predictions[0])
                confidence = predictions[0][predicted_digit]
                
                # Display in col2
                with col2:
                    st.subheader("Prediction Results")
                    st.metric("Predicted Digit", predicted_digit)
                    st.metric("Confidence", f"{confidence*100:.2f}%")
                    
                    # Bar chart of probabilities
                    st.bar_chart(predictions[0])
                    
                    # Show all probabilities
                    st.write("**All Probabilities:**")
                    for digit in range(10):
                        st.write(f"Digit {digit}: {predictions[0][digit]*100:.2f}%")

# Tab 2: Upload Image
with tab2:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Upload an image")
        uploaded_file = st.file_uploader(
            "Choose an image...", 
            type=['png', 'jpg', 'jpeg'],
            key="upload"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file).convert('L')
            st.image(image, caption='Uploaded Image', use_column_width=True)
            
            if st.button("🎯 Predict", key="predict_upload"):
                # Preprocess
                img = image.resize((28, 28))
                img_array = np.array(img)
                
                # Invert if needed
                if np.mean(img_array) > 127:
                    img_array = 255 - img_array
                
                img_array = img_array.astype('float32') / 255.0
                img_array = img_array.reshape(1, 28, 28, 1)
                
                # Predict
                predictions = model.predict(img_array, verbose=0)
                predicted_digit = np.argmax(predictions[0])
                confidence = predictions[0][predicted_digit]
                
                # Display results
                with col2:
                    st.subheader("Prediction Results")
                    st.metric("Predicted Digit", predicted_digit)
                    st.metric("Confidence", f"{confidence*100:.2f}%")
                    
                    # Show preprocessed image
                    st.image(img_array.reshape(28, 28), 
                            caption='Preprocessed (28x28)', 
                            width=200)
                    
                    # Bar chart
                    st.bar_chart(predictions[0])
                    
                    # All probabilities
                    st.write("**All Probabilities:**")
                    for digit in range(10):
                        st.write(f"Digit {digit}: {predictions[0][digit]*100:.2f}%")

# Sidebar info
with st.sidebar:
    st.header("ℹ️ About")
    st.write("""
    This app uses a trained CNN model to classify handwritten digits (0-9).
    
    **Features:**
    - Draw digits on canvas
    - Upload digit images
    - Real-time predictions
    - Confidence scores
    
    **Model Info:**
    - Architecture: CNN
    - Dataset: MNIST
    - Input: 28x28 grayscale
    """)
    
    st.header("📊 Model Stats")
    total_params = sum([np.prod(w.shape) for w in model.get_weights()])
    st.metric("Total Parameters", f"{total_params:,}")
    st.metric("Input Shape", "28×28×1")
    st.metric("Output Classes", "10")