import torch

def main():
    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if cuda_available:
        # Get GPU information
        gpu_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(current_device)
        
        print(f"Number of GPUs: {gpu_count}")
        print(f"Current GPU: {current_device}")
        print(f"GPU Name: {gpu_name}")

if __name__ == "__main__":
    main()
