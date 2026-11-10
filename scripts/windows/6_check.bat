@echo off
title PX1 check misses and labels
cd /d F:\Coding\Research
python -m pip install scipy > logs_setup3.txt 2>&1
python -u check_misses_labels.py --data data --model results\unet_seed0.pt --out diagnosis > logs_check.txt 2>&1
echo CHECK_EXIT %errorlevel% >> logs_check.txt
