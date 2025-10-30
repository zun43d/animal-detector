# Vercel Deployment Guide for Skin Disease Detection App

## 📁 Project Structure

```
animal-detector/
├── api/
│   └── index.py          # Flask app (Vercel serverless function)
├── static/               # Static assets
│   ├── logo.png
│   ├── graph.jpeg
│   └── ...
├── templates/            # HTML templates
│   ├── index.html
│   ├── image.html
│   └── video.html
├── Weights/              # Model weights (133MB+)
│   └── unet_swin_classification_model.weights.h5
├── vercel.json          # Vercel configuration
├── .vercelignore        # Files to ignore
├── requirements.txt     # Python dependencies
└── README.md
```

## ⚠️ Important Limitations

### 1. **File Size Limit**

- Vercel has a **50MB deployment size limit**
- Your model file (`unet_swin_classification_model.weights.h5`) is likely **larger than 50MB**
- **This will cause deployment to FAIL**

### 2. **Cold Start Issues**

- TensorFlow takes 10-20 seconds to load on cold starts
- First request will timeout (10s function timeout on free tier)

### 3. **Memory Limit**

- Free tier: 1024MB RAM
- TensorFlow + model may exceed this

## 🚀 Deployment Steps

### Option A: Deploy with External Model Storage (Recommended)

1. **Upload model to cloud storage:**

   ```bash
   # Option 1: AWS S3
   aws s3 cp Weights/unet_swin_classification_model.weights.h5 s3://your-bucket/

   # Option 2: Google Cloud Storage
   gsutil cp Weights/unet_swin_classification_model.weights.h5 gs://your-bucket/

   # Option 3: Hugging Face Hub (free for public models)
   pip install huggingface_hub
   huggingface-cli upload your-username/skin-disease-model ./Weights/
   ```

2. **Update `api/index.py` to download model on startup:**

   ```python
   import requests
   MODEL_URL = "https://your-storage-url/unet_swin_classification_model.weights.h5"
   # Download and cache model
   ```

3. **Deploy to Vercel:**
   ```bash
   vercel --prod
   ```

### Option B: Use Smaller Model

1. **Quantize your model** (reduce size):

   ```python
   import tensorflow as tf
   converter = tf.lite.TFLiteConverter.from_keras_model(model)
   converter.optimizations = [tf.lite.Optimize.DEFAULT]
   tflite_model = converter.convert()
   ```

2. **Deploy quantized model** (should be < 50MB)

### Option C: Deploy to Railway/Render (Alternative)

These platforms have larger deployment limits:

- **Railway**: 1GB deployment size, better for ML models
- **Render**: 1GB deployment size
- **Hugging Face Spaces**: Unlimited, free for public apps

## 📝 Vercel Deployment Commands

```bash
# Install Vercel CLI globally (if not already)
npm i -g vercel

# Login to Vercel
vercel login

# Deploy (development)
vercel

# Deploy (production)
vercel --prod

# Check deployment logs
vercel logs

# Check deployment size
vercel inspect
```

## 🔧 Troubleshooting

### Error: "Deployment size exceeds limit"

**Solution**: Move model to cloud storage or use alternative platform

### Error: "Function timeout"

**Solution**:

- Upgrade to Pro plan ($20/month) for 60s timeout
- Or use Railway/Render instead

### Error: "Memory limit exceeded"

**Solution**: Use lighter model or upgrade plan

## 🎯 Recommended Approach

**For your project, I recommend:**

1. **Use Railway or Render** instead of Vercel:

   ```bash
   # Railway
   railway login
   railway init
   railway up

   # Render
   # Just connect your GitHub repo at render.com
   ```

2. **If you must use Vercel:**
   - Store model on Hugging Face Hub (free)
   - Download on first request and cache in `/tmp`
   - Be aware of 10s timeout on free tier

## 📊 Platform Comparison

| Feature    | Vercel Free | Railway Free | Render Free | HF Spaces |
| ---------- | ----------- | ------------ | ----------- | --------- |
| Size Limit | 50MB        | 1GB          | 1GB         | No limit  |
| Timeout    | 10s         | 500s         | None        | None      |
| RAM        | 1GB         | 512MB        | 512MB       | 16GB      |
| GPU        | ❌          | ❌           | ❌          | ✅ (paid) |
| Best For   | APIs        | ML Apps      | ML Apps     | ML Demos  |

## 🚫 Why Vercel May Not Work

Your `Weights/` folder alone is likely **100MB+**, which exceeds Vercel's 50MB limit. Even with external storage, the cold start time for TensorFlow will cause timeouts.

**Verdict**: Use Railway, Render, or Hugging Face Spaces instead! 🎯
