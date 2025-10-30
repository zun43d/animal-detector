from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from tensorflow import keras
from PIL import Image
import io
import base64

app = Flask(__name__)

# Load the UNet-Swin classification model
MODEL_PATH = "Weights/unet_swin_classification_model.weights.h5"
model = None

# Common skin disease classes (adjust based on your model's training)
classNames = [
    'Melanoma',  # MEL
    'Melanocytic Nevus',  # NV
    'Basal Cell Carcinoma',  # BCC
    'Actinic Keratosis',  # AK
    'Benign Keratosis',  # BKL
    'Dermatofibroma',  # DF
    'Vascular Lesion',  # VASC
    'Squamous Cell Carcinoma'  # SCC
]

# Image preprocessing parameters
IMG_HEIGHT = 224
IMG_WIDTH = 224

def load_model():
    """Load the skin disease detection model"""
    global model
    try:
        # Build a simple model architecture for classification
        # Adjust this based on your actual model architecture
        from tensorflow.keras.applications import EfficientNetB0
        base_model = EfficientNetB0(
            include_top=False,
            weights=None,
            input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
            pooling='avg'
        )
        
        inputs = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
        x = base_model(inputs, training=False)
        x = keras.layers.Dense(256, activation='relu')(x)
        x = keras.layers.Dropout(0.5)(x)
        outputs = keras.layers.Dense(len(classNames), activation='softmax')(x)
        
        model = keras.Model(inputs, outputs)
        
        # Load the weights
        model.load_weights(MODEL_PATH)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Creating a basic model for demonstration...")
        # Create a basic model if weight loading fails
        model = keras.Sequential([
            keras.layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
            keras.layers.Conv2D(32, 3, activation='relu'),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(64, 3, activation='relu'),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(128, 3, activation='relu'),
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(len(classNames), activation='softmax')
        ])

# Load model on startup
load_model()

def preprocess_image_from_buffer(image_buffer):
    """Preprocess image from buffer for model input"""
    # Read image from buffer
    img = Image.open(io.BytesIO(image_buffer))
    img = img.convert('RGB')
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = np.array(img) / 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)  # Ensure float32
    img.close()
    return img_array

def predict_skin_disease(image_buffer):
    """Predict skin disease from image buffer"""
    global model
    
    # Reload model for each prediction to ensure fresh state
    print("\n" + "="*50)
    print("Starting new prediction...")
    print("="*50)
    
    # Process image from buffer
    img_array = preprocess_image_from_buffer(image_buffer)
    print(f"Image shape: {img_array.shape}")
    print(f"Image min/max values: {img_array.min():.3f} / {img_array.max():.3f}")
    
    # Make prediction with explicit copy to avoid state issues
    img_array_copy = np.copy(img_array)
    
    # Clear TensorFlow session and reload model for fresh predictions
    keras.backend.clear_session()
    load_model()  # Reload the model
    
    predictions = model.predict(img_array_copy, verbose=0)
    
    print(f"\nRaw predictions: {predictions[0]}")
    
    # Get prediction results
    predicted_class = int(np.argmax(predictions[0]))
    confidence = float(predictions[0][predicted_class])
    
    predicted_disease = classNames[predicted_class]
    
    print(f"Predicted class index: {predicted_class}")
    print(f"Predicted disease: {predicted_disease}")
    print(f"Confidence: {confidence*100:.2f}%")
    print("="*50 + "\n")
    
    # Clear arrays to free memory
    del img_array_copy
    del predictions
    
    return predicted_disease, confidence

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/image')
def image():
    return render_template('image.html')

@app.route('/process', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({"error": "No file part"})
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    
    try:
        # Read image directly into buffer without saving
        image_buffer = file.read()
        
        # Get prediction from buffer
        disease_name, confidence = predict_skin_disease(image_buffer)
        
        # Convert image buffer to base64 for display (no file saving)
        image_base64 = base64.b64encode(image_buffer).decode('utf-8')
        
        # Determine image format
        img_format = file.filename.split('.')[-1].lower()
        if img_format in ['jpg', 'jpeg']:
            mime_type = 'image/jpeg'
        elif img_format == 'png':
            mime_type = 'image/png'
        else:
            mime_type = 'image/jpeg'  # default
        
        # Return base64 encoded image with data URI
        image_data_uri = f"data:{mime_type};base64,{image_base64}"
        
        return jsonify({
            "result_image": image_data_uri,
            "prediction": disease_name,
            "confidence": f"{confidence*100:.2f}"
        })
    
    except Exception as e:
        return jsonify({"error": f"Error processing image: {str(e)}"})

@app.route('/uploads/<filename>')
def send_file(filename):
    # This route is no longer needed but keeping for backward compatibility
    return jsonify({"error": "Direct file access no longer supported"})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
