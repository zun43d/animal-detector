# Skin Disease Detection System

An AI-powered web application for detecting and classifying skin diseases using deep learning. This Flask-based application uses a UNet-Swin Transformer model to identify seven different types of skin lesions from uploaded images.

## Features

- **AI-Powered Detection**: Uses a UNet-Swin classification model for accurate skin disease detection
- **8 Disease Categories**:
  - Melanoma (MEL)
  - Melanocytic Nevus (NV)
  - Basal Cell Carcinoma (BCC)
  - Actinic Keratosis (AK)
  - Benign Keratosis (BKL)
  - Dermatofibroma (DF)
  - Vascular Lesion (VASC)
  - Squamous Cell Carcinoma (SCC)
- **User-Friendly Interface**: Clean, responsive web interface built with TailwindCSS
- **Real-time Processing**: Upload images and get instant predictions
- **Confidence Scores**: Displays prediction confidence for transparency

## Installation

### Prerequisites

Make sure you have Python 3.8+ installed on your system.

### Install System Dependencies

```bash
sudo apt-get update && sudo apt-get install ffmpeg libsm6 libxext6
```

### Install Python Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Run the Application

```bash
python app.py
```

The application will start on `http://localhost:5000`

### Using the Web Interface

1. Navigate to `http://localhost:5000` in your web browser
2. Click on "Model" to access the detection page
3. Upload an image of a skin lesion or capture one using your camera
4. View the prediction results with confidence scores

## Model Information

- **Architecture**: UNet-Swin Transformer
- **Framework**: TensorFlow/Keras
- **Input Size**: 224x224 RGB images
- **Model File**: `Weights/unet_swin_classification_model.weights.h5`

## Important Disclaimer

⚠️ **This tool is for educational and screening purposes only.**

The predictions made by this system should NOT be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified dermatologist or healthcare provider for proper medical evaluation and treatment of any skin condition.

## Project Structure

```
.
├── app.py                  # Main Flask application
├── requirements.txt        # Python dependencies
├── README.md              # Project documentation
├── Weights/
│   └── unet_swin_classification_model.weights.h5  # Model weights
├── templates/
│   ├── index.html         # Home page
│   └── image.html         # Detection page
├── static/                # Static assets (images, etc.)
└── uploads/               # Uploaded and processed images (created at runtime)
```

## Technology Stack

- **Backend**: Flask (Python web framework)
- **ML Framework**: TensorFlow/Keras
- **Image Processing**: OpenCV, PIL
- **Frontend**: HTML, TailwindCSS, DaisyUI
- **Model Architecture**: UNet-Swin Transformer

## Development

To modify the model or add new features:

1. Update the model architecture in `app.py` if needed
2. Ensure the model weights match your architecture
3. Adjust the `classNames` list if you have different disease categories
4. Modify preprocessing parameters (`IMG_HEIGHT`, `IMG_WIDTH`) as needed

## License

This project is for educational purposes.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.
