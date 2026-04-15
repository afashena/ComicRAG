from pathlib import Path
import gc
import torch

from captioner.generate_caption_qwen import load_qwen, caption_image
from captioner.finetuning.finetune import load_finetuned_model, SYSTEM_PROMPT

def evaluate_model_on_test_images(test_dir: Path, model, processor, output_dir: Path, finetuned: bool):
    """
    Evaluate model on test images and save captions to text files.
    
    Args:
        test_dir: Directory containing test images
        model: Model to evaluate
        processor: Processor for the model
        output_dir: Directory to save results (baseline/ or finetuned/)
        finetuned: Whether this is a finetuned model
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    for img_path in test_dir.glob("*.JPG"):
        print(f"\nEvaluating image: {img_path.name}")

        # Generate caption
        caption = caption_image(
            image_path=str(img_path),
            model=model,
            processor=processor,
            system_prompt=SYSTEM_PROMPT,
            user_prompt="Caption this comic panel.",
            schema=None,
        )

        # Save caption to text file
        output_file = output_dir / f"{img_path.stem}.txt"
        with open(output_file, "w") as f:
            f.write(caption)

        if finetuned:
            print(f"Finetuned model caption:\n{caption}\n")
            print(f"Saved to: {output_file}")
        else:
            print(f"Baseline Caption:\n{caption}\n")
            print(f"Saved to: {output_file}")

if __name__ == "__main__":
    # run test images through the baseline model and then the finetuned model, to compare outputs side by side
    test_dir = Path("./test_split_images")
    adapter_path = Path("./lora_output/final_adapter")
    eval_results_dir = Path(__file__).parent / "eval_results"
    eval_results_dir.mkdir(parents=True, exist_ok=True)
    
    baseline_dir = eval_results_dir / "baseline"
    finetuned_dir = eval_results_dir / "finetuned"

    print("\n" + "="*60)
    print("EVALUATING BASELINE MODEL")
    print("="*60)
    baseline_model, baseline_processor = load_qwen()
    evaluate_model_on_test_images(test_dir, baseline_model, baseline_processor, baseline_dir, finetuned=False)

    # Unload baseline model from memory
    print("\nUnloading baseline model from memory...")
    del baseline_model
    del baseline_processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Baseline model unloaded. GPU memory cleared.\n")

    print("="*60)
    print("EVALUATING FINETUNED MODEL")
    print("="*60)
    finetuned_model, finetuned_processor = load_finetuned_model(adapter_path=adapter_path)
    evaluate_model_on_test_images(test_dir, finetuned_model, finetuned_processor, finetuned_dir, finetuned=True)
    
    print("\n" + "="*60)
    print(f"Results saved to: {eval_results_dir.absolute()}")
    print("="*60)

