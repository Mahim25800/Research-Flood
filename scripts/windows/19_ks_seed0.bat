@echo off
title PX1 Kuro Siwo test, model seed 0 redo (waits) - do not close
cd /d F:\Coding\Research
echo [%date% %time%] waiting for model seeds 1-2 of the Kuro Siwo test >> logs_ks_seed0.txt
:wait
if exist runs_ks\ks_summary_m2_v2.md goto go
timeout /t 300 /nobreak > nul
goto wait
:go
echo [%date% %time%] model seed 0, merged events >> logs_ks_seed0.txt
python -u ks_eval.py --data data --ks data_ks --runs runs --model-seed 0 --out runs_ks >> logs_ks_seed0.txt 2>&1
if errorlevel 1 (
  echo   crashed, resuming >> logs_ks_seed0.txt
  python -u ks_eval.py --data data --ks data_ks --runs runs --model-seed 0 --out runs_ks >> logs_ks_seed0.txt 2>&1
)
echo [%date% %time%] KS_SEED0_V2_DONE >> logs_ks_seed0.txt
