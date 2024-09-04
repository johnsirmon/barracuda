import torch
import matplotlib.pyplot as plt
import psutil

def display_gpu_info():
    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if not cuda_available:
        print("CUDA is not available on this system.")
        return

    # Get number of GPUs
    gpu_count = torch.cuda.device_count()
    print(f"Number of GPUs: {gpu_count}")
    
    # Prepare lists to store GPU information
    gpu_names = []
    total_memories = []
    allocated_memories = []
    cached_memories = []
    
    for i in range(gpu_count):
        # Get GPU name
        gpu_name = torch.cuda.get_device_name(i)
        gpu_names.append(gpu_name)
        
        # Get memory stats
        total_memory = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)  # Convert to GB
        allocated_memory = torch.cuda.memory_allocated(i) / (1024 ** 3)  # Convert to GB
        cached_memory = torch.cuda.memory_reserved(i) / (1024 ** 3)  # Convert to GB
        
        total_memories.append(total_memory)
        allocated_memories.append(allocated_memory)
        cached_memories.append(cached_memory)

        print(f"GPU {i}: {gpu_name}")
        print(f"  - Total Memory: {total_memory:.2f} GB")
        print(f"  - Allocated Memory: {allocated_memory:.2f} GB")
        print(f"  - Cached Memory: {cached_memory:.2f} GB")

    # Plot GPU memory usage
    fig, ax = plt.subplots()
    width = 0.2  # Width of the bars
    x = range(gpu_count)
    
    ax.bar(x, total_memories, width, label='Total Memory (GB)', color='skyblue')
    ax.bar([i + width for i in x], allocated_memories, width, label='Allocated Memory (GB)', color='orange')
    ax.bar([i + 2 * width for i in x], cached_memories, width, label='Cached Memory (GB)', color='green')
    
    ax.set_xlabel('GPU Index')
    ax.set_title('GPU Memory Usage')
    ax.set_xticks([i + width for i in x])
    ax.set_xticklabels([f'GPU {i}' for i in x])
    ax.legend()

    plt.show()

def display_cuda_torch_info():
    # CUDA Version used by PyTorch
    cuda_version = torch.version.cuda
    print(f"CUDA Version (PyTorch): {cuda_version}")

    # PyTorch version
    torch_version = torch.__version__
    print(f"PyTorch Version: {torch_version}")

    # Check cuDNN version
    cudnn_version = torch.backends.cudnn.version()
    print(f"cuDNN Version: {cudnn_version}")

def display_system_info():
    # Get basic system info
    cpu_count = psutil.cpu_count(logical=True)
    memory_info = psutil.virtual_memory()

    print(f"CPU Count: {cpu_count}")
    print(f"Total System Memory: {memory_info.total / (1024 ** 3):.2f} GB")

def main():
    print("=== System Information ===")
    display_system_info()
    print("\n=== CUDA and PyTorch Information ===")
    display_cuda_torch_info()
    print("\n=== GPU Information ===")
    display_gpu_info()

if __name__ == "__main__":
    main()
