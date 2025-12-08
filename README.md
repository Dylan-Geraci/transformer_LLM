# transformer_LLM

## Below is how to install the major dependencies if you don't go through requirements.txt (NOT RECOMMENDED)

# 1) make sure pip up-to-date
python -m pip install --upgrade pip

# 2) remove old torch builds (so pip won't try to reuse incompatible wheel)
pip uninstall -y torch torchvision torchaudio

# 3) Install CUDA-enabled PyTorch (CUDA 12.1 builds). This is the official index URL.
pip install --index-url https://download.pytorch.org/whl/cu121 \
    "torch" "torchvision" "torchaudio"

# 4) Install Hugging Face + PEFT + helpers
pip install --upgrade transformers accelerate peft datasets \
    scikit-learn PyPDF2 safetensors

# 5) Optional: if you rely on tokenizers' fast Rust backend
pip install --upgrade tokenizers

# 6) install SentencePiece + tokenizers (fast tokenizer backend)
pip install --upgrade sentencepiece tokenizers

# 7) (optional but safe) install HF transformers with the sentencepiece extra
pip install --upgrade "transformers[sentencepiece]" safetensors