@echo off
title Kuro Siwo download - do not close
cd /d F:\Coding\Research
echo [%date% %time%] start >> logs_ks_download.txt
set n=0
:retry
python -u ks_download.py --splits test_GRD train_GRD --step-mb 512 --k 6 >> logs_ks_download.txt 2>&1
if not errorlevel 1 goto done
set /a n+=1
echo [%date% %time%] crashed, resuming (retry %n%) >> logs_ks_download.txt
timeout /t 60 /nobreak > nul
if %n% lss 5 goto retry
:done
echo [%date% %time%] KS_EXIT %errorlevel% >> logs_ks_download.txt
