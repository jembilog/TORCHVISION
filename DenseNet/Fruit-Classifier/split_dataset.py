import os
import random
import shutil

#paths
SOURCE_DIR = "dataset/FRUITS/images"
OUTPUT_DIR = "dataset"

TRAIN_SPLIT = 0.80
VAL_SPLIT = 0.10
TEST_SPLIT = 0.10

random.seed(42)

#create folder
for split in ["train", "validation", "test"]:
    os.makedirs(os.path.join(OUTPUT_DIR, split), exist_ok=True)

#process each class
for class_name in os.listdir(SOURCE_DIR):
    class_path = os.path.join(SOURCE_DIR, class_name)

    if not os.path.isdir(class_path):
        continue

    images = [
        img for img in os.listdir(class_path)
        if img.lower().endswith((".png",".jpg",".jpeg"))
    ]

    random.shuffle(images)

    total = len(images)
    train_size = int(total * TRAIN_SPLIT)
    val_size = int(total * VAL_SPLIT)

    train_images = images[:train_size]
    val_images = images[train_size:train_size + val_size]
    test_images = images[train_size + val_size:]

    for split in ["train", "validation", "test"]:
        os.makedirs(
            os.path.join(OUTPUT_DIR, split, class_name),
            exist_ok=True
        )

    #copy train
    for img in train_images:
        shutil.copy(
            os.path.join(class_path,img),
            os.path.join(OUTPUT_DIR,"train",class_name,img)
        )

    #copy val
    for img in val_images:
        shutil.copy(
            os.path.join(class_path,img),
            os.path.join(OUTPUT_DIR,"validation",class_name, img)
        )

    #copy test
    for img in test_images:
        shutil.copy(
            os.path.join(class_path, img),
            os.path.join(OUTPUT_DIR, "test", class_name, img)
        )
    print(
        f"{class_name}: "
        f"Train={len(train_images)}, "
        f"Validation={len(val_images)}, "
        f"Test={len(test_images)}"
    )

print("\nDataset splitting completed successfully!") 
