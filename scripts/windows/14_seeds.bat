@echo off
title PX1 repeat runs - do not close
cd /d F:\Coding\Research
echo [%date% %time%] SEEDS START >> logs_seeds.txt
call 9_status.bat
type logs_status.txt >> logs_seeds.txt

rem 1) repeat the adaptation on the existing models with two more adaptation seeds (about 70 min)
call :tta 0 1
call :tta 0 2
python -u tta_seeds_summary.py > logs_seeds_summary.txt 2>&1

rem 2) retrain all 22 leave-one-flood-out models with seeds 1 and 2 (about 5.3 h each), adapting each set
for %%S in (1 2) do (
  for %%E in (Bolivia Ghana India Mekong Nigeria Pakistan Paraguay Somalia Spain Sri-Lanka USA) do (
    call :train V0 %%E %%S --inputs s1
    call :train V1 %%E %%S --inputs s1dem
  )
  call :tta %%S 0
  python -u tta_seeds_summary.py > logs_seeds_summary.txt 2>&1
)
echo [%date% %time%] SEEDS_DONE >> logs_seeds.txt
exit /b 0

:tta
echo [%date% %time%] adapt models seed %1, adaptation seed %2 >> logs_seeds.txt
python -u tta_px1.py --data data --runs runs_loeo --model-seed %1 --seed %2 >> runs_loeo\log_tta_m%1_a%2.txt 2>&1
if errorlevel 1 (
  echo   crashed, resuming >> logs_seeds.txt
  python -u tta_px1.py --data data --runs runs_loeo --model-seed %1 --seed %2 >> runs_loeo\log_tta_m%1_a%2.txt 2>&1
)
echo   exit %errorlevel% >> logs_seeds.txt
goto :eof

:train
echo [%date% %time%] %1 hold %2 seed %3 >> logs_seeds.txt
if exist runs_loeo\%1_hold%2_seed%3\results.json (
  echo   already done, skipping >> logs_seeds.txt
  goto :eof
)
python -u train_px1.py --data data --variant %1 --holdout %2 --seed %3 --workers 0 --out runs_loeo %4 %5 >> runs_loeo\log_%1_hold%2_seed%3.txt 2>&1
if errorlevel 1 (
  echo   crashed, retrying from last checkpoint >> logs_seeds.txt
  python -u train_px1.py --data data --variant %1 --holdout %2 --seed %3 --workers 0 --out runs_loeo %4 %5 >> runs_loeo\log_%1_hold%2_seed%3.txt 2>&1
)
echo   exit %errorlevel% >> logs_seeds.txt
goto :eof
