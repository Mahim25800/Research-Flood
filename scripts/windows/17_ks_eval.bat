@echo off
title PX1 Kuro Siwo test (waits for download) - do not close
cd /d F:\Coding\Research
echo [%date% %time%] waiting for the Kuro Siwo download and the extra adaptation seeds >> logs_ks_eval.txt
:wait
findstr /c:"KS_DOWNLOAD_DONE" logs_ks_download.txt > nul 2>&1
if errorlevel 1 goto pause
findstr /c:"hold Bolivia seed 1" logs_seeds.txt > nul 2>&1
if errorlevel 1 goto pause
goto go
:pause
timeout /t 300 /nobreak > nul
goto wait
:go
echo [%date% %time%] start >> logs_ks_eval.txt
call :one 0
call :one 1
call :one 2
echo [%date% %time%] KS_EVAL_ALL_DONE >> logs_ks_eval.txt
exit /b 0

:one
echo [%date% %time%] model seed %1 >> logs_ks_eval.txt
python -u ks_eval.py --data data --ks data_ks --runs runs --model-seed %1 --out runs_ks >> logs_ks_eval.txt 2>&1
if errorlevel 1 (
  echo   crashed, resuming >> logs_ks_eval.txt
  python -u ks_eval.py --data data --ks data_ks --runs runs --model-seed %1 --out runs_ks >> logs_ks_eval.txt 2>&1
)
goto :eof
