# requirements:
# uv add transformers>=4.49.0 peft trl datasets bitsandbytes accelerate
# uv add pillow torch torchvision

import json
from pathlib import Path
from dotenv import load_dotenv

import torch
from peft import get_peft_model
from PIL import Image
from pydantic import BaseModel, Field

from trl import SFTTrainer
import matplotlib.pyplot as plt
from transformers import TrainerCallback, EarlyStoppingCallback

from config import Config
from captioner.model import QwenCaptioner
from captioner.finetuning.train_utils import get_training_test_splits

def load_config(config_path: Path) -> dict:
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        return config_dict

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
#MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
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


class QwenTrainer(QwenCaptioner):
    def __init__(self, config: Config, system_prompt: str, user_prompt: str, schema: BaseModel | None):
        super().__init__(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema=schema,
            config=config,
        )

    def make_collator(self):
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

                messages = self.get_structured_prompt(image=image, gt_label=example["label"])

                text = self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,  # False during training
                )
                texts.append(text)

            # Process all images and texts together
            batch = self.processor(
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

    def train(
            self,
        ):
        # Print useful system info
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        if device == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"VRAM free: {torch.cuda.mem_get_info()[0] / 1024**3:.1f} GB")

        # ── Preprocess and load dataset ──────────────────────────────────────────────────────────
        train_dataset, eval_dataset = get_training_test_splits(Path(self.config.train_dir))

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

        self.load_qwen()

        self.model.config.use_cache = False  # required for gradient checkpointing

        # ── LoRA config ───────────────────────────────────────────────────────────
        lora_config = config.build_lora()

        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

        # ── Training config ───────────────────────────────────────────────────────
        sft_config = config.build_sft()

        loss_callback = LossPlotCallback(output_dir=self.config.sft_config["output_dir"])

        # ── Trainer ───────────────────────────────────────────────────────────────
        trainer = SFTTrainer(
            model=self.model,
            args=sft_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=self.make_collator(),
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
        adapter_path = Path(self.config.sft_config["output_dir"]) / "final_adapter"
        self.model.save_pretrained(adapter_path)
        self.processor.save_pretrained(adapter_path)
        print(f"\nLoRA adapter saved to {adapter_path}")
        print("Training complete.")



# ── 7. Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    load_dotenv()

    config = Config.model_validate(load_config(Path("./config.json")))

    system_prompt = (
            "You are a comic panel captioning assistant for golden age Western comics. "
            "Analyze the panel and respond without making up any details not evident in the image. "
        )
    
    user_prompt = ("Without knowing or making up narrative elements which are not in the panel, "
                    "describe the setting, characters, and events evident in this comic panel as if you were telling a prose narrative.")
    
    captioner_trainer = QwenTrainer(config=config, system_prompt=system_prompt, user_prompt=user_prompt, schema=None)
    captioner_trainer.train()