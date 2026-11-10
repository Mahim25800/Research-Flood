@echo off
title PX1 pipeline v2 - do not close
cd /d F:\Coding\Research
echo [%date% %time%] stopping old pipeline > logs_restart.txt
powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*7_run_all.bat*' -or $_.CommandLine -like '*train_px1.py*' } | ForEach-Object { 'stopping ' + $_.ProcessId + ' ' + $_.Name; Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }" >> logs_restart.txt 2>&1
timeout /t 5 /nobreak > nul
powershell -NoProfile -Command "$o = Get-CimInstance Win32_OperatingSystem; 'RAM total GB: ' + [math]::Round($o.TotalVisibleMemorySize/1MB,1) + ', free GB: ' + [math]::Round($o.FreePhysicalMemory/1MB,1) + ', commit free GB: ' + [math]::Round($o.FreeVirtualMemory/1MB,1)" >> logs_restart.txt 2>&1
echo [%date% %time%] RESTART (training only, audit kept) >> logs_all.txt
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
python -u train_px1.py --data data --variant %1 --seed %2 --workers 0 %3 %4 %5 %6 > runs\log_%1_seed%2.txt 2>&1
echo   exit %errorlevel% >> logs_all.txt
goto :eof
