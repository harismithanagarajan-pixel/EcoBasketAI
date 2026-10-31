import os
from pathlib import Path

BASE_DIR = Path("database/dataset")

datasets = {
    "Cake": "cake",
    "Cheese": "cheese"
}

extensions = (".jpg", ".jpeg", ".png", ".webp")

for name, folder in datasets.items():
    print("\n" + "=" * 40)
    print(name.upper())
    print("=" * 40)

    for split in ["train", "test", "valid"]:
        folder_path = BASE_DIR / folder / split

        if folder_path.exists():
            count = sum(
                1 for file in folder_path.rglob("*")
                if file.is_file() and file.suffix.lower() in extensions
            )
            print(f"{split}: {count} images")
        else:
            print(f"{split}: Folder not found")

print("\nDataset verification completed!")