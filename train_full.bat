@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul 2>&1

REM ===================================================================
REM  LoABot v5.9 - Alamet Temporal YZ Egitim Pipeline
REM ===================================================================
REM
REM  ESKi (v5.8):  JPEG manifest + AGENTIC_LOGS -> 3ch siniflandirma
REM  YENi (v5.9):  MP4 video + JSONL            -> 9ch koordinat regresyon
REM
REM  Pipeline:
REM    [0/4] CUDA kontrolu
REM    [1/4] video_dataset_builder.py --dry-run  (ayiklama/dogrulama)
REM    [2/4] TensorBoard baslatma
REM    [3/4] train_agentic.py                    (temporal egitim)
REM
REM  Veri Yapisi:
REM    VIDEO_ROOT\
REM    +-- VID_20260301_143000\
REM    |   +-- video.mp4          <- H.264 ekran kaydi
REM    |   +-- actions.jsonl      <- Tiklama/tus loglari
REM    |   +-- session_meta.json  <- Oturum metadatasi
REM ===================================================================

REM === Klasor Yollari ===
set "VIDEO_ROOT=D:\LoABot_Training_Data\videos"
set "TRAIN_LOG_ROOT=D:\LoABot_Training_Data\runtime_data\training_logs"

REM === Egitim Parametreleri ===
set "EPOCHS=60"
set "BACKBONE=resnet18"
set "LOSS_FN=mse"
set "IMAGE_SIZE=224"
set "LR=0.0001"
set "TRAIN_RATIO=0.8"
set "PATIENCE=8"
set "BATCH_SIZE=24"
set "NUM_WORKERS=1"

REM === Python Yolu ===
set "PYTHON=C:\LoABot\.venv\Scripts\python.exe"
if not exist "%PYTHON%" (
    set "PYTHON=python"
)

echo.
echo  =============================================
echo   LoABot v5.9 Alamet Temporal YZ Egitimi
echo   Mimari: 9-Kanal Stacked 2D CNN [Regresyon]
echo  =============================================
echo.

REM ===================================================================
REM  [0/4] CUDA DOGRULAMA
REM ===================================================================

echo [0/4] CUDA kontrolu yapiliyor...
"%PYTHON%" -c "import torch; ok=torch.cuda.is_available(); dev=torch.cuda.get_device_name(0) if ok else 'YOK'; vram=round(torch.cuda.get_device_properties(0).total_memory/1024**3,1) if ok else 0; print(f'  PyTorch: {torch.__version__} | CUDA: {torch.version.cuda} | GPU: {dev} | VRAM: {vram} GB'); exit(0 if ok else 2)"

if %ERRORLEVEL% EQU 2 (
    echo.
    echo  [HATA] CUDA bulunamadi! GPU destekli PyTorch gerekli.
    echo.
    echo  Cozum:
    echo    %PYTHON% -m pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu124
    echo.
    pause
    exit /b 2
)
echo.

REM ===================================================================
REM  [1/4] VIDEO DATASET DOGRULAMA (Ayiklama Modulu / Dry-Run)
REM ===================================================================
REM
REM  video_dataset_builder.py --dry-run:
REM    - VID_* klasorlerini tarar, video.mp4 + actions.jsonl eslestirir
REM    - Bozuk video (decord acilamaz) veya eksik JSONL tespit eder
REM    - 3 frame'den az video varsa otomatik atlar
REM    - Oturum sayisi, aksiyon dagilimi, kaynak (bot/user) istatistikleri
REM    - DataLoader OLUSTURMAZ (sadece dogrulama + istatistik)
REM    - Hata varsa exit code != 0 doner, egitim BASLAMAZ
REM

echo [1/4] Video dataset dogrulamasi yapiliyor (dry-run)...
echo       Kaynak: %VIDEO_ROOT%
echo.

"%PYTHON%" "video_dataset_builder.py" ^
  --video-root "%VIDEO_ROOT%" ^
  --train-ratio %TRAIN_RATIO% ^
  --dry-run

set "VALIDATE_CODE=%ERRORLEVEL%"

if %VALIDATE_CODE% NEQ 0 (
    echo.
    echo  =============================================
    echo  [HATA] Dataset dogrulama BASARISIZ! Kod: %VALIDATE_CODE%
    echo.
    echo  Olasi nedenler:
    echo    - %VIDEO_ROOT% klasoru bos veya mevcut degil
    echo    - VID_* klasorlerinde video.mp4 veya actions.jsonl eksik
    echo    - actions.jsonl bozuk JSON iceriyor
    echo    - decord yuklu degil: pip install decord
    echo.
    echo  Once video kayitlarinin dogru toplandigini kontrol edin.
    echo  =============================================
    pause
    exit /b %VALIDATE_CODE%
)

echo.
echo  [OK] Dataset dogrulama basarili. Egitim baslatiliyor...
echo.

REM ===================================================================
REM  [2/4] TENSORBOARD
REM ===================================================================

echo [2/4] TensorBoard baslatiliyor (port 6006)...
start /b cmd /c "tensorboard --logdir="%TRAIN_LOG_ROOT%" --port=6006 >nul 2>&1"
echo       http://localhost:6006
echo.

REM ===================================================================
REM  [3/4] TEMPORAL EGITIM
REM ===================================================================
REM
REM  train_agentic.py v2.0 (Temporal):
REM    Girdi  : [T-1, T, T+1] -> torch.cat -> (B, 9, H, W)
REM    Model  : ResNet18/EfficientNet-B0 (ilk conv 3ch -> 9ch)
REM    Cikti  : Sigmoid -> (B, 2) normalize koordinat [0, 1]
REM    Loss   : MSELoss (sadece mouse_click orneklerinde)
REM    Metrik : val_mse_loss, val_mean_dist (Oklid mesafesi)
REM
REM  --batch-size ve --num-workers guvenli sabit degerlerle geciliyor:
REM    decode tarafindaki ffmpeg/decord bellek sivramalarini azaltir.
REM

echo [3/4] Alamet Temporal YZ egitimi basliyor...
echo  ------------------------------------------
echo   Video Root  : %VIDEO_ROOT%
echo   Backbone    : %BACKBONE% (9-kanal temporal)
echo   Loss        : %LOSS_FN%
echo   Epochs      : %EPOCHS%
echo   LR          : %LR%
echo   Image Size  : %IMAGE_SIZE%x%IMAGE_SIZE%
echo   Train Ratio : %TRAIN_RATIO%
echo   Patience    : %PATIENCE% (early stopping)
echo   Batch Size  : %BATCH_SIZE%
echo   Num Workers : %NUM_WORKERS%
echo  ------------------------------------------
echo.

"%PYTHON%" "train_agentic.py" ^
  --video-root "%VIDEO_ROOT%" ^
  --log-root "%TRAIN_LOG_ROOT%" ^
  --backbone %BACKBONE% ^
  --loss-fn %LOSS_FN% ^
  --image-size %IMAGE_SIZE% ^
  --epochs %EPOCHS% ^
  --lr %LR% ^
  --train-ratio %TRAIN_RATIO% ^
  --early-stop-patience %PATIENCE% ^
  --batch-size %BATCH_SIZE% ^
  --num-workers %NUM_WORKERS%

set "TRAIN_CODE=%ERRORLEVEL%"

REM ===================================================================
REM  [4/4] SONUC
REM ===================================================================

echo.
echo  =============================================
if "%TRAIN_CODE%"=="0" (
    echo   [OK] Alamet'in beyni basariyla oruldu.
    echo.
    echo   Checkpoint : %TRAIN_LOG_ROOT%\run_*\best.pt
    echo   Metrikler  : %TRAIN_LOG_ROOT%\run_*\metrics.csv
    echo   TensorBoard: %TRAIN_LOG_ROOT%\run_*\tensorboard\
    echo.
    echo   Canliya almak icin:
    echo     copy best.pt models\active_model\best.pt
) else (
    echo   [HATA] Egitim sirasinda sorun cikti. Kod: %TRAIN_CODE%
    echo.
    echo   Olasi nedenler:
    echo     - VRAM yetersiz
    echo     - decord yuklu degil: pip install decord
    echo     - Video dosyalari bozuk veya cok kisa
)
echo  =============================================
echo.

pause
exit /b %TRAIN_CODE%
