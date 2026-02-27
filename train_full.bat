@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM === Klasor Yollari ===
set "DATA_ROOT=E:\LoABot_Training_Data"
set "DATASETS_ROOT=%DATA_ROOT%\datasets"
set "TRAIN_MANIFEST=%DATASETS_ROOT%\vision\vision_train.jsonl"
set "TEST_MANIFEST=%DATASETS_ROOT%\vision\vision_test.jsonl"
set "ACTION_LOG_ROOT=%DATA_ROOT%\AGENTIC_LOGS"
set "TRAIN_LOG_ROOT=%DATA_ROOT%\runtime_data\training_logs"

REM === Epoch sayisi (batch/worker otomatik ayarlanir) ===
set "EPOCHS=60"
set "PYTHON=C:\LoABot\.venv\Scripts\python.exe"

REM Python kontrolu
if not exist "%PYTHON%" (
    set "PYTHON=python"
)

REM === [0/3] CUDA Dogrulama ===
echo [0/3] CUDA kontrolu yapiliyor...
"%PYTHON%" -c "import torch; ok=torch.cuda.is_available(); dev=torch.cuda.get_device_name(0) if ok else 'YOK'; vram=round(torch.cuda.get_device_properties(0).total_memory/1024**3,1) if ok else 0; print(f'PyTorch: {torch.__version__} | CUDA: {torch.version.cuda} | GPU: {dev} | VRAM: {vram} GB'); exit(0 if ok else 2)"

if %ERRORLEVEL% EQU 2 (
    echo.
    echo [HATA] CUDA bulunamadi! .venv icindeki PyTorch CPU-only kurulu olmali.
    echo Asagidaki komutla GPU destekli PyTorch yukleyin:
    echo.
    echo   %PYTHON% -m pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu124
    echo.
    echo Yukledikten sonra bu dosyayi tekrar calistirin.
    pause
    exit /b 2
)

echo [1/3] TensorBoard baslatiliyor...
start /b cmd /c "tensorboard --logdir="%TRAIN_LOG_ROOT%" --port=6006 >nul 2>&1"

echo [2/3] Alamet Eylemci YZ egitimi basliyor...
echo ------------------------------------------
echo Veri Seti: 16,933 gorsel hazir.
echo Hedef: %EPOCHS% Epoch  ^|  Batch ve Worker: Otomatik (GPU/CPU'ya gore)
echo ------------------------------------------

REM batch-size ve num-workers gecilmiyor; train_agentic.py GPU/CPU'ya gore otomatik ayarlar.
"%PYTHON%" "train_agentic.py" ^
  --train-manifest "%TRAIN_MANIFEST%" ^
  --test-manifest "%TEST_MANIFEST%" ^
  --action-log-root "%ACTION_LOG_ROOT%" ^
  --log-root "%TRAIN_LOG_ROOT%" ^
  --epochs %EPOCHS%

set "EXIT_CODE=%ERRORLEVEL%"

echo.
echo ------------------------------------------
if "%EXIT_CODE%"=="0" (
    echo [OK] Alamet'in beyni basariyla oruldu.
) else (
    echo [HATA] Egitim sirasinda sorun cikti. Kod: %EXIT_CODE%
)
echo ------------------------------------------

pause
exit /b %EXIT_CODE%