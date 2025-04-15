import torch
import platform
import sys

print(f"--- System Information ---")
print(f"Platform: {platform.platform()}")
print(f"Python Version: {sys.version}")
print(f"PyTorch Version: {torch.__version__}")
print("-" * 25)

print(f"--- CUDA Availability ---")
cuda_available = torch.cuda.is_available()
print(f"CUDA Available: {cuda_available}")

if cuda_available:
    gpu_count = torch.cuda.device_count()
    print(f"Number of GPUs available: {gpu_count}")

    current_device_id = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(current_device_id)
    print(f"Current GPU ID: {current_device_id}")
    print(f"Current GPU Name: {gpu_name}")

    major, minor = torch.cuda.get_device_capability(current_device_id)
    print(f"CUDA Capability: {major}.{minor}")

    total_mem_bytes = torch.cuda.get_device_properties(current_device_id).total_memory
    total_mem_gb = round(total_mem_bytes / (1024**3), 2)
    print(f"Total GPU Memory: {total_mem_gb} GB")
else:
    print("CUDA is not available. PyTorch will run on CPU.")
    print("Possible Reasons:")
    print("- NVIDIA drivers are not installed or up-to-date.")
    print("- Compatible CUDA toolkit is not installed.")
    print("- PyTorch was installed without CUDA support (e.g., CPU-only version). Check installation command.")

print("-" * 25)