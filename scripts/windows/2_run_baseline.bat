@echo off
cd /d F:\Coding\Research
title PX1 baseline - do not close
echo [%date% %time%] setup > logs_setup.txt
powershell -NoProfile -Command "Get-PSDrive F | Format-Table Used,Free -AutoSize" >> logs_setup.txt 2>&1
python -m pip install rasterio >> logs_setup.txt 2>&1
python -c "import rasterio;print('rasterio',rasterio.__version__)" >> logs_setup.txt 2>&1
echo SETUP_DONE >> logs_setup.txt
echo Downloading Sen1Floods11...
python -u download_sen1floods11.py --out data > logs_download.txt 2>&1
if errorlevel 1 (
  echo DOWNLOAD_FAILED >> logs_download.txt
  exit /b 1
)
echo DOWNLOAD_DONE >> logs_download.txt
echo Training U-Net on GPU...
python -u sen1floods11_unet_baseline.py --data data --epochs 60 --batch 8 --out results > logs_train.txt 2>&1
echo TRAIN_EXIT %errorlevel% >> logs_train.txt
