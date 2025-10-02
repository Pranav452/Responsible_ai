
import torch
from transformers import TrainingArguments
from datasets import load_dataset
from trl import SFTTrainer
import yaml
import argparse
import os

# Import the key Unsloth library
from unsloth import FastLanguageModel
import wandb

def main(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print("✅ Loading base model and tokenizer with Unsloth...")
    
    max_seq_length = 1024 # We can adjust this if needed
    
    # This uses Unsloth's pre-quantized model for maximum reliability
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/llama-3-8b-bnb-4bit",
        max_seq_length = max_seq_length,
        dtype = None,
        load_in_4bit = True,
    )

    print("✅ Configuring LoRA adapters with Unsloth...")
    model = FastLanguageModel.get_peft_model(
        model,
        r = config['lora_r'],
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha = config['lora_alpha'],
        lora_dropout = config['lora_dropout'],
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
    )

    print("✅ Loading and preparing the training dataset...")
    dataset = load_dataset(config['dataset_name'], split = "train")
    train_dataset = dataset.shuffle(seed=42).select(range(1500))

    print("✅ Setting up Training Arguments and Initializing SFTTrainer...")
    
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        dataset_text_field = "chosen",
        max_seq_length = max_seq_length,
        packing = False,
        args = TrainingArguments(
            per_device_train_batch_size = config['per_device_train_batch_size'],
            gradient_accumulation_steps = config['gradient_accumulation_steps'],
            warmup_steps = 5,
            num_train_epochs = config['num_train_epochs'],
            learning_rate = config['learning_rate'],
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "outputs", # Temporary output dir
            report_to = "wandb",
        ),
    )
    
    # Initialize W&B
    wandb.init(project=config['wandb_project'], name="qlora-llama3-unsloth", config=trainer.args.to_dict())

    print("\n--- Starting Training ---")
    trainer.train()
    print("--- Training Complete ---\n")
    
    wandb.finish()

    print(f"✅ Saving final LoRA adapter to {config['output_dir']}...")
    trainer.model.save_pretrained(config['output_dir'])
    tokenizer.save_pretrained(config['output_dir'])
    print("✅ Adapter saved successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/training_config.yaml")
    args = parser.parse_args()
    main(args.config)
