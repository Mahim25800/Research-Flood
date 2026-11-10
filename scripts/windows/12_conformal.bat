@echo off
title PX1 miss-rate guarantee
cd /d F:\Coding\Research
echo [%date% %time%] start > logs_conformal.txt
python -u conformal_px1.py --data data --runs runs_loeo >> logs_conformal.txt 2>&1
echo [%date% %time%] CONFORMAL_EXIT %errorlevel% >> logs_conformal.txt
