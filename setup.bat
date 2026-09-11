@echo off
echo ============================================
echo  Diffusion LM - Environment Setup
echo ============================================
echo.

echo Installing Python dependencies...
pip install accelerate datasets tokenizers transformers tqdm numpy einops imageio pillow flask tiktoken
echo.

echo ============================================
echo  Verifying GPU setup...
echo ============================================
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'; print('GPU:', gpu); print('BF16:', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False); vram = torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0; print(f'VRAM: {vram:.1f} GB')"
echo.

echo ============================================
echo  Setup complete!
echo ============================================
echo.
echo Next steps:
echo   1. Download FineWeb dataset from Kaggle:
echo      https://www.kaggle.com/datasets/abdulwahidrukua/fineweb-memmap-tokens
echo   2. Extract .bin files to: data\fineweb\
echo   3. Run: python pretrain.py
echo.
pause
