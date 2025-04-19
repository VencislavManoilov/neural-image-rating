import torch
import sys

def check_cuda():
    print("PyTorch version:", torch.__version__)
    
    if not torch.cuda.is_available():
        print("\n❌ CUDA is not available. Issues to check:")
        print("  1. NVIDIA GPU driver may not be installed correctly")
        print("  2. PyTorch was not installed with CUDA support")
        print("\nTo install PyTorch with CUDA support, run:")
        print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print("\nNVIDIA driver installation guide: https://www.nvidia.com/Download/index.aspx")
        return False
    
    print("\n✅ CUDA is available!")
    print("CUDA version:", torch.version.cuda)
    print("Device count:", torch.cuda.device_count())
    
    for i in range(torch.cuda.device_count()):
        print(f"\nDevice {i}:")
        print(f"  Name: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"  Total memory: {props.total_memory / 1024**3:.2f} GB")
        
    # Test CUDA tensor operations
    print("\nTesting CUDA Tensor operations...")
    try:
        x = torch.rand(1000, 1000).cuda()
        y = torch.rand(1000, 1000).cuda()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        z = torch.mm(x, y)
        end.record()
        
        torch.cuda.synchronize()
        print(f"Matrix multiplication time: {start.elapsed_time(end):.2f} ms")
        print("✅ CUDA operations working correctly!")
        return True
    except Exception as e:
        print(f"❌ CUDA operation failed: {e}")
        return False

if __name__ == "__main__":
    success = check_cuda()
    sys.exit(0 if success else 1)
