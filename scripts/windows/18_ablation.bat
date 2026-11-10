@echo off
title PX1 physics-rule ablation - do not close
cd /d F:\Coding\Research
echo [%date% %time%] start >> logs_ablation.txt
set n=0
:retry
python -u tta_px1.py --data data --runs runs_loeo --seed 0 --model-seed 0 --methods phys_drop,phys_add >> logs_ablation.txt 2>&1
if not errorlevel 1 goto done
set /a n+=1
echo [%date% %time%] crashed, resuming (retry %n%) >> logs_ablation.txt
if %n% lss 3 goto retry
:done
echo [%date% %time%] ABLATION_EXIT %errorlevel% >> logs_ablation.txt
