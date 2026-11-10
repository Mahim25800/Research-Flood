@echo off
title PX1 leave-one-event-out - do not close
cd /d F:\Coding\Research
if not exist runs_loeo mkdir runs_loeo
echo [%date% %time%] LOEO START >> logs_loeo.txt
call 9_status.bat
type logs_status.txt >> logs_loeo.txt
for %%E in (Bolivia Ghana India Mekong Nigeria Pakistan Paraguay Somalia Spain Sri-Lanka USA) do (
  call :run V0 %%E --inputs s1
  call :run V1 %%E --inputs s1dem
)
echo [%date% %time%] summary >> logs_loeo.txt
python -u summarize_loeo.py > logs_loeo_summary.txt 2>&1
echo [%date% %time%] LOEO_DONE >> logs_loeo.txt
exit /b 0

:run
echo [%date% %time%] %1 hold %2 >> logs_loeo.txt
if exist runs_loeo\%1_hold%2_seed0\results.json (
  echo   already done, skipping >> logs_loeo.txt
  goto :eof
)
python -u train_px1.py --data data --variant %1 --holdout %2 --seed 0 --workers 0 --out runs_loeo %3 %4 >> runs_loeo\log_%1_hold%2.txt 2>&1
if errorlevel 1 (
  echo   crashed, retrying from last checkpoint >> logs_loeo.txt
  python -u train_px1.py --data data --variant %1 --holdout %2 --seed 0 --workers 0 --out runs_loeo %3 %4 >> runs_loeo\log_%1_hold%2.txt 2>&1
)
echo   exit %errorlevel% >> logs_loeo.txt
goto :eof
