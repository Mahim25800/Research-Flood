@echo off
title PX1 flood adaptation - do not close
cd /d F:\Coding\Research
echo [%date% %time%] start > logs_tta.txt
call 9_status.bat
type logs_status.txt >> logs_tta.txt
set n=0
:retry
python -u tta_px1.py --data data --runs runs_loeo >> logs_tta.txt 2>&1
if not errorlevel 1 goto done
set /a n+=1
echo [%date% %time%] crashed, resuming (retry %n%) >> logs_tta.txt
if %n% lss 3 goto retry
:done
echo [%date% %time%] TTA_EXIT %errorlevel% >> logs_tta.txt
