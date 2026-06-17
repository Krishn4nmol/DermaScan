@echo off
REM ─────────────────────────────────────────────────────────────────────────
REM  Skin Lesion Detection — One-Click Setup for Windows
REM  Run this from inside the skin_lesion/ folder
REM ─────────────────────────────────────────────────────────────────────────

echo ============================================================
echo   Skin Lesion Detection — Environment Setup
echo ============================================================
echo.

REM Check conda is available
where conda >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] conda not found. Please install Anaconda or Miniconda first.
    pause
    exit /b 1
)

echo [1/4] Creating conda environment (skin_lesion, Python 3.11) ...
call conda create -n skin_lesion python=3.11 -y
if %errorlevel% neq 0 goto error

echo.
echo [2/4] Activating environment ...
call conda activate skin_lesion
if %errorlevel% neq 0 goto error

echo.
echo [3/4] Installing PyTorch (CPU version - works on all AMD hardware) ...
call pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
if %errorlevel% neq 0 goto error

echo.
echo [4/4] Installing remaining dependencies ...
call pip install -r requirements.txt
if %errorlevel% neq 0 goto error

echo.
echo ============================================================
echo   Setup Complete!
echo ============================================================
echo.
echo Next steps:
echo.
echo   1. Download HAM10000 from Kaggle:
echo      https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
echo.
echo   2. Place these files in data\raw\:
echo      - HAM10000_images_part_1.zip
echo      - HAM10000_images_part_2.zip
echo      - HAM10000_metadata.csv
echo.
echo   3. Run:  conda activate skin_lesion
echo            python prepare_data.py
echo            python train.py --model mobilenetv3 --workers 0
echo.
echo ============================================================
pause
exit /b 0

:error
echo.
echo [ERROR] Setup failed at the step above.
echo Please check the error message and try again.
pause
exit /b 1
