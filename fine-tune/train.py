import os
import types
from mlx_lm import lora

# Paths
MLX_DATA_DIR = "./fine-tune/mlx-data"
OUTPUT_DIR = "./persisted/"

def main():
    # Ensure the output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Model name
    model_name = "mlx-community/Meta-Llama-3-8B-Instruct-4bit"
    
    # Configuration for MLX-LM Lora training
    # Optimized for lower memory usage on Mac
    training_config = {
        "model": model_name,
        "train": True,
        "data": MLX_DATA_DIR,
        "iters": 600,            # Restored for full retraining
        "batch_size": 1,         # Reduced from 4 to minimize memory usage
        "learning_rate": 1e-5,
        "steps_per_report": 10,
        "steps_per_eval": 100,
        "save_every": 100,
        "adapter_path": OUTPUT_DIR,
        "max_seq_length": 1024,  # Reduced from 2048 to prevent OOM
        "grad_checkpoint": True, # Enabled to save memory
        "grad_accumulation_steps": 4, # Increased to maintain effective batch size
        "lora_parameters": {"rank": 8, "dropout": 0.0, "scale": 20.0}, # Reduced rank
        # Defaults
        "test": False,
        "test_batches": 500,
        "val_batches": 10,      # Reduced from 25
        "seed": 0,
        "resume_adapter_file": None,
        "mask_prompt": False,
        "report_to": None,
        "project_name": None,
        "optimizer": "adam",
        "optimizer_config": {"adam": {}, "adamw": {}, "muon": {}, "sgd": {}, "adafactor": {}},
        "num_layers": 16,
        "lr_schedule": None,
        "fine_tune_type": "lora"
    }
    
    print(f"Starting training on Apple Silicon (MPS) with model: {model_name}...")
    print("Memory-optimized settings: batch_size=1, max_seq_length=1024, grad_checkpoint=True")
    
    # Create a namespace object as expected by lora.run
    args = types.SimpleNamespace(**training_config)
    
    # Execute training
    lora.run(args)
    
    print(f"Training complete. Adapters saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
