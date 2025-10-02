from datasets import load_dataset
import os

def main():
    print("Loading Anthropic/hh-rlhf dataset...")
    dataset = load_dataset("Anthropic/hh-rlhf", split="test")
    
    # Using 1000 examples for a robust evaluation set
    eval_dataset = dataset.shuffle(seed=42).select(range(1000))
    
    output_path = "src/data/evaluation_set.jsonl"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Fixed: Use correct method for saving to JSONL
    eval_dataset.to_json(output_path)
    print(f"✅ Evaluation set with {len(eval_dataset)} examples saved.")

if __name__ == "__main__":
    main()