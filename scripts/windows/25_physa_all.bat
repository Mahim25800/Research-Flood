@echo off
title PX1 adaptive physics runs (waits for HAND) - do not close
cd /d F:\Coding\Research
echo [%date% %time%] waiting for the HAND download >> logs_physa.txt
:wait
findstr /c:"HAND_DONE" logs_hand.txt > nul 2>&1
if not errorlevel 1 goto verify
timeout /t 120 /nobreak > nul
goto wait
:verify
echo [%date% %time%] checking for blank HAND chips >> logs_physa.txt
python -u hand_fetch.py --data data --ks data_ks --workers 8 --verify >> logs_physa.txt 2>&1
echo [%date% %time%] starting four runs >> logs_physa.txt
start "PX1 adaptive physics, Sen1Floods11" cmd /c "python -u tta_px1.py --data data --runs runs_loeo --seed 0 --model-seed 0 --methods physa,adabn_physa >> logs_physa_s1f.txt 2>&1 & echo PHYSA_S1F_EXIT >> logs_physa_s1f.txt"
start "PX1 adaptive physics, Kuro Siwo model 1" cmd /c "python -u ks_eval.py --data data --ks data_ks --runs runs --model-seed 0 --out runs_ks --methods physa,adabn_physa >> logs_physa_ks0.txt 2>&1 & echo PHYSA_KS_EXIT >> logs_physa_ks0.txt"
start "PX1 adaptive physics, Kuro Siwo model 2" cmd /c "python -u ks_eval.py --data data --ks data_ks --runs runs --model-seed 1 --out runs_ks --methods physa,adabn_physa >> logs_physa_ks1.txt 2>&1 & echo PHYSA_KS_EXIT >> logs_physa_ks1.txt"
start "PX1 adaptive physics, Kuro Siwo model 3" cmd /c "python -u ks_eval.py --data data --ks data_ks --runs runs --model-seed 2 --out runs_ks --methods physa,adabn_physa >> logs_physa_ks2.txt 2>&1 & echo PHYSA_KS_EXIT >> logs_physa_ks2.txt"
echo [%date% %time%] launched >> logs_physa.txt
