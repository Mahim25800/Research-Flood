@echo off
title PX1 resume
cd /d F:\Coding\Research
taskkill /FI "WINDOWTITLE eq PX1 baseline - do not close*" /T /F > logs_resume.txt 2>&1
timeout /t 3 /nobreak > nul
del /q data\S1Hand\*.part data\LabelHand\*.part data\flood_handlabeled\*.part 2>nul
echo Resuming download...
python -u download_sen1floods11.py --out data > logs_download.txt 2>&1
if errorlevel 1 (
  echo DOWNLOAD_FAILED >> logs_download.txt
  exit /b 1
)
echo DOWNLOAD_DONE >> logs_download.txt
echo Training U-Net on GPU...
python -u sen1floods11_unet_baseline.py --data data --epochs 60 --batch 8 --out results > logs_train.txt 2>&1
echo TRAIN_EXIT %errorlevel% >> logs_train.txt
