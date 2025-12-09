#!/usr/bin/env python3
"""
check_environment.py

Environment validation script for Transformer LLM Assignment.
Checks Python version, installed packages, and GPU availability.
"""

import sys
import subprocess


def check_python_version():
    """Check if Python version is 3.9+"""
    version = sys.version_info
    print(f"Python version: {sys.version.split()[0]}")

    if version >= (3, 9):
        print("  [OK] Python 3.9+ requirement met")
        return True
    else:
        print(f"  [FAIL] Python 3.9+ required, found {version.major}.{version.minor}")
        return False


def check_gpu():
    """Check NVIDIA GPU availability"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode == 0:
            gpu_info = result.stdout.strip()
            print(f"\nGPU detected: {gpu_info}")
            return True
        else:
            print("\n[FAIL] nvidia-smi failed")
            return False

    except FileNotFoundError:
        print("\n[FAIL] nvidia-smi not found (no NVIDIA GPU or drivers not installed)")
        return False
    except subprocess.TimeoutExpired:
        print("\n[FAIL] nvidia-smi timeout")
        return False


def check_module(module_name, import_name=None):
    """Check if a module is installed and importable"""
    if import_name is None:
        import_name = module_name

    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"  [OK] {module_name}: {version}")
        return True
    except ImportError:
        print(f"  [FAIL] {module_name}: NOT INSTALLED")
        return False


def check_pytorch_cuda():
    """Check PyTorch CUDA support"""
    try:
        import torch
        print(f"\nPyTorch: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  GPU count: {torch.cuda.device_count()}")

            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                props = torch.cuda.get_device_properties(i)
                memory_gb = props.total_memory / (1024 ** 3)
                print(f"  GPU {i}: {name} ({memory_gb:.2f} GB)")

            # Test GPU allocation
            try:
                x = torch.randn(100, 100, device='cuda')
                y = torch.randn(100, 100, device='cuda')
                z = torch.matmul(x, y)
                print(f"  [OK] GPU computation test passed")
                del x, y, z
                torch.cuda.empty_cache()
                return True
            except Exception as e:
                print(f"  [FAIL] GPU computation test failed: {e}")
                return False
        else:
            print("  [WARN] CUDA not available - will use CPU (much slower)")
            return False

    except ImportError:
        print("[FAIL] PyTorch not installed")
        return False


def main():
    print("="*70)
    print("TRANSFORMER LLM ASSIGNMENT - ENVIRONMENT VALIDATION")
    print("="*70)

    # Check Python version
    python_ok = check_python_version()

    # Check required modules
    print("\nRequired Python Packages:")
    modules = {
        'torch': 'torch',
        'transformers': 'transformers',
        'peft': 'peft',
        'datasets': 'datasets',
        'accelerate': 'accelerate',
        'numpy': 'numpy',
        'scikit-learn': 'sklearn',
        'PyPDF2': 'PyPDF2',
        'safetensors': 'safetensors',
        'sentencepiece': 'sentencepiece',
        'tokenizers': 'tokenizers',
    }

    modules_ok = []
    for display_name, import_name in modules.items():
        ok = check_module(display_name, import_name)
        modules_ok.append(ok)

    all_modules_ok = all(modules_ok)

    # Check GPU
    gpu_ok = check_gpu()

    # Check PyTorch CUDA
    pytorch_cuda_ok = check_pytorch_cuda()

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    if python_ok and all_modules_ok:
        print("[OK] All required packages installed")
    else:
        print("[FAIL] Some packages missing")
        print("  Run: pip install -r requirements.txt")

    if gpu_ok and pytorch_cuda_ok:
        print("[OK] GPU detected and PyTorch CUDA working")
        print("  Ready for GPU-accelerated training")
    elif gpu_ok and not pytorch_cuda_ok:
        print("[WARN] GPU detected but PyTorch CUDA not working")
        print("  Reinstall PyTorch with CUDA support:")
        print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124")
    else:
        print("[WARN] No GPU detected - will use CPU")
        print("  Training will be significantly slower")
        print("  Consider using --skip-heavy flag with main.py")

    print("\n" + "="*70)

    # Exit code
    if python_ok and all_modules_ok:
        print("Environment is ready!")
        sys.exit(0)
    else:
        print("Environment has issues - please fix before proceeding")
        sys.exit(1)


if __name__ == "__main__":
    main()
