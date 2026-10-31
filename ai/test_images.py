import os
from PIL import Image

dataset_path = "database/dataset"

valid_count = 0
invalid_count = 0

for category in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, category)

    if os.path.isdir(folder_path):
        for filename in os.listdir(folder_path):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                image_path = os.path.join(folder_path, filename)

                try:
                    img = Image.open(image_path)
                    img.verify()
                    valid_count += 1

                except Exception:
                    print("INVALID:", category, filename)
                    invalid_count += 1

print("\nImage Validation Completed!")
print("Valid Images:", valid_count)
print("Invalid Images:", invalid_count)