import os

dataset_path = "database/dataset"

categories = ["cake", "cheese"]

for category in categories:

    print("\n==========", category.upper(), "==========")

    for split in ["train", "test", "valid"]:

        folder_path = os.path.join(dataset_path, category, split)

        print("\n", split.upper(), ":")

        if not os.path.exists(folder_path):
            print("Folder not found!")
            continue

        items = os.listdir(folder_path)

        if not items:
            print("Empty folder!")
            continue

        for item in items:

            item_path = os.path.join(folder_path, item)

            if os.path.isdir(item_path):
                print("Subfolder:", item)

            else:
                print("File:", item)

print("\nChecking completed!")