import torch
import os


def print_status(check_name, success, message=""):
    """Helper function to print formatted status messages."""
    status = "✅ PASS" if success else "❌ FAIL"
    print(f"{check_name:<50} [{status}] {message}")
    if not success:
        # Print extra debugging info for the most critical check
        if "bitsandbytes" in check_name:
            print("\n--- BITSANDBYTES DEBUG INFO ---")
            os.system("python -m bitsandbytes")
            print("---------------------------------\n")
        raise SystemExit(f"Verification failed at: {check_name}")


def verify_environment():
    """
    Runs a comprehensive check of all critical components for the project.
    """
    print("\n" + "=" * 60)
    print("🔬 RUNNING ENVIRONMENT PRE-FLIGHT CHECK 🔬")
    print("=" * 60 + "\n")

    # --- Check 1: PyTorch and GPU ---
    try:
        import torch

        torch_version = torch.__version__
        gpu_available = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if gpu_available else "N/A"
        print_status(
            "PyTorch Import & GPU Availability",
            gpu_available,
            f"(PyTorch: {torch_version}, GPU: {gpu_name})",
        )
    except Exception as e:
        print_status("PyTorch Import & GPU Availability", False, f"Error: {e}")

    # --- Check 2: bitsandbytes CUDA Backend ---
    try:
        import bitsandbytes as bnb

        # The ultimate test: try to instantiate a 4-bit linear layer on the GPU.
        dummy_layer = torch.nn.Linear(10, 10, bias=False).cuda()
        quantized_layer = bnb.nn.Linear4bit(
            dummy_layer.in_features, dummy_layer.out_features, bias=False
        ).cuda()
        print_status(
            "bitsandbytes GPU Backend (QLoRA)",
            True,
            "Successfully created a 4-bit layer on GPU.",
        )
    except Exception as e:
        print_status("bitsandbytes GPU Backend (QLoRA)", False, f"Critical Error: {e}")

    # --- Check 3: Core Hugging Face Libraries ---
    try:
        import transformers, peft, trl, accelerate, datasets

        print_status(
            "Hugging Face Ecosystem (transformers, peft, trl, etc.)",
            True,
            "All libraries imported successfully.",
        )
    except Exception as e:
        print_status("Hugging Face Ecosystem", False, f"Error: {e}")

    # --- Check 4: Evaluation Libraries ---
    try:
        print("Loading detoxify model (this may take a moment)...")
        import detoxify

        # Try loading the model to ensure all its dependencies (like torch) are met.
        detoxify_model = detoxify.Detoxify("original")
        import wandb

        print_status(
            "Evaluation & Logging (detoxify, wandb)",
            True,
            "All libraries imported successfully.",
        )
    except Exception as e:
        print_status("Evaluation & Logging", False, f"Error: {e}")

    print("\n" + "=" * 60)
    print("🎉 ALL CHECKS PASSED! Environment is ready for the mission. 🎉")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    verify_environment()
