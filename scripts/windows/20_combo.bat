@echo off
title PX1 recalibrate + physics (Sen1Floods11) - do not close
cd /d F:\Coding\Research
echo [%date% %time%] start >> logs_combo.txt
set n=0
:retry
python -u tta_px1.py --data data --runs runs_loeo --seed 0 --model-seed 0 --methods adabn,adabn_phys,tent_phys >> logs_combo.txt 2>&1
if not errorlevel 1 goto done
set /a n+=1
echo [%date% %time%] crashed, resuming (retry %n%) >> logs_combo.txt
if %n% lss 3 goto retry
:done
echo [%date% %time%] COMBO_EXIT %errorlevel% >> logs_combo.txt
