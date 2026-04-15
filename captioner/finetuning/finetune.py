# requirements:
# uv add transformers>=4.49.0 peft trl datasets bitsandbytes accelerate
# uv add pillow torch torchvision

import json
import shutil
from pathlib import Path
from typing import Optional
from enum import Enum
from dotenv import load_dotenv

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from PIL import Image
from pydantic import BaseModel, Field
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer
import matplotlib.pyplot as plt
from transformers import TrainerCallback, EarlyStoppingCallback

from captioner.generate_caption_qwen import load_qwen  # reusing the same model loading code
from config import Config

class LossPlotCallback(TrainerCallback):
    """
    Collects training and validation loss at every logging/eval step
    and saves a plot when training ends.
    """

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.train_steps  = []
        self.train_losses = []
        self.eval_steps   = []
        self.eval_losses  = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called at every logging_steps — captures training loss."""
        if logs is None:
            return
        if "loss" in logs:
            self.train_steps.append(state.global_step)
            self.train_losses.append(logs["loss"])

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Called after every evaluation run — captures validation loss."""
        if metrics is None:
            return
        if "eval_loss" in metrics:
            self.eval_steps.append(state.global_step)
            self.eval_losses.append(metrics["eval_loss"])

    def on_train_end(self, args, state, control, **kwargs):
        """Called once when training finishes — saves the plot."""
        self._save_plot()

    def _save_plot(self):
        if not self.train_losses:
            print("No loss values collected — skipping plot.")
            return

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(
            self.train_steps,
            self.train_losses,
            label="Training loss",
            color="steelblue",
            linewidth=1.5,
        )

        if self.eval_losses:
            ax.plot(
                self.eval_steps,
                self.eval_losses,
                label="Validation loss",
                color="coral",
                linewidth=1.5,
                marker="o",       # dot at each eval point so it's easy to read
                markersize=4,
            )

        ax.set_xlabel("Training step")
        ax.set_ylabel("Loss")
        ax.set_title("Training and validation loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Mark the best checkpoint step if available
        if self.eval_losses:
            best_idx  = self.eval_losses.index(min(self.eval_losses))
            best_step = self.eval_steps[best_idx]
            best_loss = self.eval_losses[best_idx]
            ax.axvline(
                x=best_step,
                color="coral",
                linestyle="--",
                alpha=0.5,
                label=f"Best checkpoint (step {best_step}, loss {best_loss:.4f})",
            )
            ax.legend()  # re-draw legend to include the new line

        plt.tight_layout()

        plot_path = f"{self.output_dir}/loss_plot.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"\nLoss plot saved to {plot_path}")



# ── 1. Model config — swap this block when moving to Lambda ──────────────────

# Local test (your GTX 1050)
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
MODEL_PATH = Path(r"C:\Users\BabyBunny\Documents\Models\qwen2.5-vl-3b-instruct")
USE_4BIT = True       # required on 3GB GPU

# Lambda Cloud (uncomment when ready)
# MODEL_ID = "google/gemma-4-E4B-it"
# USE_4BIT = False      # run in bfloat16 on A10 24GB — cleaner gradients
# LORA_RANK = 16
# BATCH_SIZE = 4
# GRAD_ACCUM = 4
# MAX_STEPS = -1        # -1 = run full dataset for NUM_EPOCHS epochs

class ComicPanelCaption(BaseModel):
        description: str = Field(
            description=("Without knowing or making up narrative elements which are not in the panel, describe the setting, characters, and events evident in this comic panel as if you were telling a prose narrative."))


# ── 3. Dataset loading ────────────────────────────────────────────────────────

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
    images_dir = train_dir.parent / "images"
    labels_dir = train_dir.parent / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Move image files to images/ and text files to labels/
    for chunk in train_dir.iterdir():
        for file in chunk.iterdir():
            if file.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".JPG", ".JPEG"}:
                shutil.move(str(file), str(images_dir / file.name))
            elif file.suffix.lower() == ".txt":
                shutil.move(str(file), str(labels_dir / file.name))

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


# ── 4. Collator — converts raw examples into model inputs ────────────────────

SYSTEM_PROMPT = (
    "You are a comic panel captioning assistant. "
    "Analyze the panel and respond with a JSON object matching the schema exactly. "
    "Return only valid JSON. No markdown, no explanation, no code fences."
)


def make_collator(processor, device):
    """
    Returns a collate_fn that:
    - Opens each image
    - Builds the chat message structure
    - Applies the chat template
    - Returns tensors ready for the model
    """
    def collate_fn(examples):
        images = []
        texts = []

        for example in examples:
            image = Image.open(example["image_path"]).convert("RGB")
            image.thumbnail((512, 512))
            images.append(image)

            messages = [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": "Caption this comic panel."},
                    ],
                },
                {
                    "role": "assistant",
                    # The ground truth label — what the model learns to produce
                    "content": example["label"],
                },
            ]

            text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,  # False during training
            )
            texts.append(text)

        # Process all images and texts together
        batch = processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        )

        # Labels are the same as input_ids for causal LM training
        # SFTTrainer will mask the prompt tokens automatically
        batch["labels"] = batch["input_ids"].clone()
        
        # Do NOT move to device here — let the trainer handle device placement
        # Moving tensors to GPU in collate_fn causes memory pinning conflicts
        return batch

    return collate_fn


# ── 5. Main training script ───────────────────────────────────────────────────

def load_config(config_path: Path) -> dict:
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    return config_dict

def train(
    train_dir: Path,
    config: Config,
    output_dir: str = "./lora_output",
):
    # Print useful system info
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM free: {torch.cuda.mem_get_info()[0] / 1024**3:.1f} GB")

    # ── Preprocess and load dataset ──────────────────────────────────────────────────────────
    train_dataset, eval_dataset = get_training_test_splits(train_dir)

    # # ── Load model ────────────────────────────────────────────────────────────
    # if USE_4BIT:
    #     bnb_config = BitsAndBytesConfig(
    #         load_in_4bit=True,
    #         bnb_4bit_quant_type="nf4",
    #         bnb_4bit_compute_dtype=torch.bfloat16,
    #         bnb_4bit_use_double_quant=True,
    #         #llm_int8_enable_fp32_cpu_offload=True,
    #     )
    #     model = AutoModelForImageTextToText.from_pretrained(
    #         MODEL_PATH,
    #         quantization_config=bnb_config,
    #         device_map="cuda:0",
    #         low_cpu_mem_usage=True,
    #         trust_remote_code=True,
    #     )
    # else:
    #     # Lambda Cloud path — full bfloat16, no quantization
    #     model = AutoModelForImageTextToText.from_pretrained(
    #         MODEL_PATH,
    #         torch_dtype=torch.bfloat16,
    #         device_map="cuda:0",
    #         trust_remote_code=True,
    #     ).to(device)

    model, processor = load_qwen()

    model.config.use_cache = False  # required for gradient checkpointing

    # ── LoRA config ───────────────────────────────────────────────────────────
    lora_config = config.build_lora()

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Training config ───────────────────────────────────────────────────────
    sft_config = config.build_sft()

    loss_callback = LossPlotCallback(output_dir=output_dir)

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=make_collator(processor, device),
        callbacks=[EarlyStoppingCallback(
                early_stopping_patience=5,
                early_stopping_threshold=0.001,)
                ,loss_callback],
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    print("\nStarting training...")
    trainer.train()
    loss_callback._save_plot()  # Save the loss plot at the end of training

    # ── Save LoRA adapter weights only (not the full model) ──────────────────
    adapter_path = Path(output_dir) / "final_adapter"
    model.save_pretrained(adapter_path)
    processor.save_pretrained(adapter_path)
    print(f"\nLoRA adapter saved to {adapter_path}")
    print("Training complete.")


# ── 6. Loading the fine-tuned model for inference ────────────────────────────

def load_finetuned_model(adapter_path: str):
    """
    Load the base model and apply the saved LoRA adapter for inference.
    Use this after training to test your fine-tuned model.
    """
    from peft import PeftModel

    processor = AutoProcessor.from_pretrained(adapter_path, trust_remote_code=True)

    # Use 4-bit quantization to match baseline model and save memory
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    base_model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map="cuda:0",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    # Merge LoRA adapters into the base model for faster inference
    model = model.merge_and_unload()
    
    model.eval()

    return model, processor


# ── 7. Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    load_dotenv()
    config = Config.model_validate(load_config(Path("./config.json")))
    train(
        train_dir=Path(r"C:\Users\BabyBunny\Documents\Data\test_finetune\merged_train"),
        config=config,
        output_dir="./lora_output",
    )