from pathlib import Path
import shutil
from datasets import Dataset


def get_training_test_splits(train_dir: Path, save_test_images: bool = True) -> tuple[Path, Path]:

    image_dir, label_dir = organize_train_set(train_dir)
    dataset = load_dataset_from_dir(image_dir, label_dir)

    # Train/validation split — hold out 10% for validation
    split = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")
    print(train_dataset[0])  # sanity check

    # Save test split images for inspection
    if save_test_images:
        test_images_dir = Path("./test_split_images")
        if test_images_dir.exists():
            version = 1
            while (Path(".") / f"test_split_images_{version}").exists():
                version += 1
            test_images_dir = Path(f"./test_split_images_{version}")
        test_images_dir.mkdir(parents=True, exist_ok=True)
        for i, example in enumerate(eval_dataset):
            shutil.copy(example["image_path"], test_images_dir / f"test_{i:03d}_{Path(example['image_path']).name}")
        print(f"Test split images saved to {test_images_dir.absolute()}")

    return train_dataset, eval_dataset

def organize_train_set(train_dir: Path) -> tuple[Path, Path]:
    """
    Organize your training data into the expected structure:
    train_dir/
        images/
            0_0.jpg
            0_1.jpg
            ...
        labels/
            0_0.txt
            0_1.txt
            ...
    """
    parent_dir = train_dir.parent
    
    # Find next available version for images directory
    images_dir = parent_dir / "images"
    if images_dir.exists():
        version = 1
        while (parent_dir / f"images_{version}").exists():
            version += 1
        images_dir = parent_dir / f"images_{version}"
    
    # Find next available version for labels directory
    labels_dir = parent_dir / "labels"
    if labels_dir.exists():
        version = 1
        while (parent_dir / f"labels_{version}").exists():
            version += 1
        labels_dir = parent_dir / f"labels_{version}"
    
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Move image files to images/ and text files to labels/
    for chunk in train_dir.iterdir():
        for file in chunk.iterdir():
            if file.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".JPG", ".JPEG"}:
                shutil.copy2(str(file), str(images_dir / file.name))
            elif file.suffix.lower() == ".txt":
                shutil.copy2(str(file), str(labels_dir / file.name))

    return images_dir, labels_dir

def load_dataset_from_dir(
    image_dir: str,
    label_dir: str,
) -> Dataset:
    """
    Pairs each image with its text label file.
    Expects image_dir/*.jpg (or png etc) and label_dir/*.txt
    with matching stems (e.g. 9_4.jpg <-> 9_4.txt).
    """
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".JPG", ".JPEG"}

    examples = []
    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix.lower() not in extensions:
            continue
        label_path = label_dir / (image_path.stem + ".txt")
        if not label_path.exists():
            print(f"  Warning: no label found for {image_path.name}, skipping.")
            continue
        with open(label_path) as f:
            label = f.read().strip()
        # Validate against Pydantic schema — catches label errors early
        # try:
        #     ComicPanelCaption.model_validate(label)
        # except Exception as e:
        #     print(f"  Warning: invalid label for {image_path.name}: {e}, skipping.")
        #     continue
        examples.append({
            "image_path": str(image_path),
            "label": label,
            "text": label,  # SFTTrainer expects a 'text' key
        })

    print(f"Loaded {len(examples)} valid labeled examples.")
    return Dataset.from_list(examples)