import os

# Dataset location
dataset_path = "database/dataset"

# Supported image formats
image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

total_images = 0

print("\n========================================")
print("       ECOBASKETAI DATASET REPORT")
print("========================================")

if not os.path.exists(dataset_path):
    print("ERROR: Dataset folder not found!")
    print("Expected location:", dataset_path)

else:

    # Check all folders recursively
    for root, dirs, files in os.walk(dataset_path):

        image_count = 0

        for file in files:

            if file.lower().endswith(image_extensions):
                image_count += 1

        if image_count > 0:

            folder_name = os.path.relpath(root, dataset_path)

            print(folder_name, ":", image_count, "images")

            total_images += image_count

    print("\n========================================")
    print("TOTAL DATASET IMAGES:", total_images)
    print("========================================")

print("\nDataset checking completed!")