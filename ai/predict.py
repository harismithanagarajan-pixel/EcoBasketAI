import os
import numpy as np
import tensorflow as tf
from PIL import Image
from tkinter import Tk, filedialog

# Get project directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Model path
MODEL_PATH = os.path.join(BASE_DIR, "ai", "models", "fruit_model.keras")

# Image size
IMG_SIZE = 128

# Fruit categories (must match training order)
categories = [
    "FreshApple",
    "FreshBanana",
    "FreshGrape",
    "FreshGuava",
    "FreshJujube",
    "FreshOrange",
    "FreshPomegranate",
    "FreshStrawberry",
    "RottenApple",
    "RottenBanana",
    "RottenGrape",
    "RottenGuava",
    "RottenJujube",
    "RottenOrange",
    "RottenPomegranate",
    "RottenStrawberry"
]

# Load trained model
print("\nLoading EcoBasketAI Model...")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "Trained model not found! Please train the model first."
    )

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

# Validate model output
if model.output_shape[-1] != len(categories):
    raise ValueError(
        "Model output does not match the 16 fruit categories!"
    )

print("Model loaded successfully!")

# Select fruit image
root = Tk()
root.withdraw()
root.attributes("-topmost", True)

image_path = filedialog.askopenfilename(
    title="Select a Fruit Image",
    filetypes=[
        ("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp")
    ]
)

root.destroy()

if not image_path:
    print("\nNo image selected!")
    raise SystemExit

# Read and preprocess image
try:
    image = Image.open(image_path).convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))

    image_array = np.array(image, dtype=np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

except Exception as e:
    print("Error processing image:", e)
    raise SystemExit

# Predict fruit category
print("\nAnalyzing Fruit Image...")

prediction = model.predict(image_array, verbose=0)

predicted_index = int(np.argmax(prediction[0]))
predicted_category = categories[predicted_index]

confidence = float(prediction[0][predicted_index]) * 100

# Extract fruit name and freshness
if predicted_category.startswith("Fresh"):
    freshness = "FRESH"
    fruit_name = predicted_category.replace("Fresh", "")
else:
    freshness = "ROTTEN"
    fruit_name = predicted_category.replace("Rotten", "")

# Display result
print("\n======================================")
print("       ECOBASKETAI RESULT")
print("======================================")

print("Image:", os.path.basename(image_path))
print("Fruit Name:", fruit_name)
print("Freshness:", freshness)
print("Confidence: {:.2f}%".format(confidence))

print("======================================")

if confidence < 60:
    print("Warning: Prediction confidence is low.")
    print("Please try another clear fruit image.")

print("\nPrediction Completed Successfully!")