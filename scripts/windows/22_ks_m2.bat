@echo off
title PX1 Kuro Siwo test, model 2 - do not close
cd /d F:\Coding\Research
echo [%date% %time%] start model seed 2 >> logs_ks_m2.txt
set n=0
:retry
python -u ks_eval.py --data data --ks data_ks --runs runs --model-seed 2 --out runs_ks >> logs_ks_m2.txt 2>&1
if not errorlevel 1 goto done
set /a n+=1
echo [%date% %time%] crashed, resuming (retry %n%) >> logs_ks_m2.txt
if %n% lss 3 goto retry
:done
echo [%date% %time%] KS_M2_EXIT %errorlevel% >> logs_ks_m2.txt
