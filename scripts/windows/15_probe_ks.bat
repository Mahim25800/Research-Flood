@echo off
title Kuro Siwo probe
cd /d F:\Coding\Research
python -u probe_kurosiwo.py --split test_GRD --shard 0 --mb 400 > logs_ks_probe.txt 2>&1
echo PROBE_EXIT %errorlevel% >> logs_ks_probe.txt
