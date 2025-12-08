@echo off
REM Setup script for rag_tune.py on Windows PC with GPU

echo ========================================
echo RAG_TUNE.PY - Windows Setup Script
echo ========================================
echo.

echo Step 1: Creating virtual environment...
python -m venv env
if errorlevel 1 (
    echo ERROR: Failed to create venv
    pause
    exit /b 1
)
echo [OK] Virtual environment created
echo.

echo Step 2: Activating virtual environment...
call env\Scripts\activate.bat
echo [OK] Virtual environment activated
echo.

echo Step 3: Upgrading pip...
python -m pip install --upgrade pip
echo [OK] Pip upgraded
echo.

echo Step 4: Installing NumPy 1.x (NOT 2.x)...
pip install "numpy<2.0"
if errorlevel 1 (
    echo ERROR: Failed to install numpy
    pause
    exit /b 1
)
echo [OK] NumPy 1.x installed
echo.

echo Step 5: Installing PyTorch with CUDA 12.1...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 (
    echo WARNING: Failed to install CUDA version, trying CPU version...
    pip install torch torchvision torchaudio
)
echo [OK] PyTorch installed
echo.

echo Step 6: Installing other dependencies...
pip install transformers PyPDF2 scikit-learn peft accelerate
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)
echo [OK] Dependencies installed
echo.

echo Step 7: Verifying installation...
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
echo.

echo ========================================
echo SETUP COMPLETE!
echo ========================================
echo.
echo To run rag_tune.py:
echo   1. Make sure venv is activated: env\Scripts\activate
echo   2. Run: python rag_tune.py
echo.
echo To deactivate venv later: deactivate
echo.
pause
