import os
import numpy as np
import tensorflow as tf

from PIL import Image
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "database", "dataset")
IMG_SIZE = 128

categories = [
    "FreshApple", "FreshBanana", "FreshGrape", "FreshGuava",
    "FreshJujube", "FreshOrange", "FreshPomegranate", "FreshStrawberry",
    "RottenApple", "RottenBanana", "RottenGrape", "RottenGuava",
    "RottenJujube", "RottenOrange", "RottenPomegranate", "RottenStrawberry"
]

print("\nLoading Fruit Dataset...")

X = []
y = []

for index, category in enumerate(categories):
    folder_path = os.path.join(DATASET_PATH, category)

    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Dataset folder not found: {folder_path}")

    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)

        try:
            image = Image.open(image_path).convert("RGB")
            image = image.resize((IMG_SIZE, IMG_SIZE))

            image_array = np.array(image, dtype=np.float32) / 255.0

            X.append(image_array)
            y.append(index)

        except Exception as e:
            print("Skipping invalid image:", image_path)
            print("Reason:", e)

    print(category, "completed")

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int32)

print("\nDataset Loading Completed!")
print("Image shape:", X.shape)
print("Label shape:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nDataset Splitting Completed!")
print("Training images:", len(X_train))
print("Testing images:", len(X_test))

print("\nBuilding CNN Model...")

model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(128, 128, 3)),

    tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(16, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# START MODEL TRAINING
print("\n================================")
print("STARTING CNN MODEL TRAINING")
print("================================")

history = model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

print("\nCNN Training Completed!")

# MODEL EVALUATION
print("\nEvaluating Model...")

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=1
)

print("\n================================")
print("MODEL EVALUATION RESULTS")
print("================================")

print("Test Loss:", test_loss)
print("Test Accuracy: {:.2f}%".format(test_accuracy * 100))

# SAVE TRAINED MODEL
MODEL_DIR = os.path.join(BASE_DIR, "ai", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "fruit_model.keras")

model.save(MODEL_PATH)

print("\n================================")
print("MODEL SAVED SUCCESSFULLY!")
print("================================")

print("Saved Location:", MODEL_PATH)
print("\nEcoBasketAI Model Training Finished!")