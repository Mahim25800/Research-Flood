@echo off
title PX1 full pipeline - do not close
cd /d F:\Coding\Research
if not exist runs mkdir runs
echo [%date% %time%] START > logs_all.txt
echo [%date% %time%] audit >> logs_all.txt
python -u audit_labels.py --data data --out audit > logs_audit.txt 2>&1
if errorlevel 1 (
  echo AUDIT_FAILED >> logs_all.txt
  exit /b 1
)
for %%S in (0 1 2) do (
  call :run V0 %%S --inputs s1
  call :run V0c %%S --inputs s1 --train-subset clean
  call :run V1 %%S --inputs s1dem
  call :run V2 %%S --inputs s1dem --phys 1.0
)
echo [%date% %time%] summary >> logs_all.txt
python -u summarize_px1.py > logs_summary.txt 2>&1
echo [%date% %time%] ALL_DONE >> logs_all.txt
exit /b 0

:run
echo [%date% %time%] %1 seed %2 >> logs_all.txt
if exist runs\%1_seed%2\results.json (
  echo   already done, skipping >> logs_all.txt
  goto :eof
)
python -u train_px1.py --data data --variant %1 --seed %2 %3 %4 %5 %6 > runs\log_%1_seed%2.txt 2>&1
echo   exit %errorlevel% >> logs_all.txt
goto :eof
