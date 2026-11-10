@echo off
title PX1 training - do not close
cd /d F:\Coding\Research
python -m pip install psutil > logs_setup4.txt 2>&1
echo [%date% %time%] RUN v3 (resume-capable, memory logged) >> logs_all.txt
call 9_status.bat
type logs_status.txt >> logs_all.txt
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
python -u train_px1.py --data data --variant %1 --seed %2 --workers 0 %3 %4 %5 %6 >> runs\log_%1_seed%2.txt 2>&1
if errorlevel 1 (
  echo   crashed, retrying from last checkpoint >> logs_all.txt
  python -u train_px1.py --data data --variant %1 --seed %2 --workers 0 %3 %4 %5 %6 >> runs\log_%1_seed%2.txt 2>&1
)
echo   exit %errorlevel% >> logs_all.txt
goto :eof
