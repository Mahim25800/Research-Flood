@echo off
title PX1 diagnosis
cd /d F:\Coding\Research
echo [%date% %time%] > logs_diagnose.txt
curl -s -o nul -w "AWS DEM reachable: HTTP %%{http_code}" -I "https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_N27_00_E068_00_DEM/Copernicus_DSM_COG_10_N27_00_E068_00_DEM.tif" >> logs_diagnose.txt 2>&1
echo. >> logs_diagnose.txt
python -m pip install matplotlib >> logs_setup2.txt 2>&1
echo Running diagnosis...
python -u diagnose_errors.py --data data --model results\unet_seed0.pt --out diagnosis >> logs_diagnose.txt 2>&1
echo DIAG_EXIT %errorlevel% >> logs_diagnose.txt
echo Caching elevation for the remaining chips (for later physics training)...
python -u diagnose_errors.py --data data --model results\unet_seed0.pt --out diagnosis --all-dem > logs_dem_cache.txt 2>&1
echo DEMCACHE_EXIT %errorlevel% >> logs_dem_cache.txt
