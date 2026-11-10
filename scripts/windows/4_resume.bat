@echo off
title PX1 resume 2
cd /d F:\Coding\Research
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine -like '*download_sen1floods11*' -or $_.CommandLine -like '*sen1floods11_unet_baseline*' } | ForEach-Object { 'stopping ' + $_.ProcessId; Stop-Process -Id $_.ProcessId -Force }" > logs_resume2.txt 2>&1
timeout /t 3 /nobreak > nul
del /q data\S1Hand\*.part data\LabelHand\*.part 2>nul
echo Resuming download...
python -u download_sen1floods11.py --out data > logs_download2.txt 2>&1
if errorlevel 1 (
  echo DOWNLOAD_FAILED >> logs_download2.txt
  exit /b 1
)
echo DOWNLOAD_DONE >> logs_download2.txt
echo Training U-Net on GPU...
python -u sen1floods11_unet_baseline.py --data data --epochs 60 --batch 8 --out results > logs_train.txt 2>&1
echo TRAIN_EXIT %errorlevel% >> logs_train.txt
