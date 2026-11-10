@echo off
title PX1 HAND download - do not close
cd /d F:\Coding\Research
echo [%date% %time%] start >> logs_hand.txt
set n=0
:retry
python -u hand_fetch.py --data data --ks data_ks --workers 8 >> logs_hand.txt 2>&1
if not errorlevel 1 goto done
set /a n+=1
echo [%date% %time%] crashed, resuming (retry %n%) >> logs_hand.txt
timeout /t 60 /nobreak > nul
if %n% lss 3 goto retry
:done
echo [%date% %time%] HAND_EXIT %errorlevel% >> logs_hand.txt
