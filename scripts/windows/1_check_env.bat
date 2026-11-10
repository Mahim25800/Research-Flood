@echo off
cd /d F:\Coding\Research
echo ==== %date% %time% > logs_env.txt
echo ---- python >> logs_env.txt
where python >> logs_env.txt 2>&1
python --version >> logs_env.txt 2>&1
where py >> logs_env.txt 2>&1
py -0p >> logs_env.txt 2>&1
echo ---- pip packages >> logs_env.txt
python -m pip list 2>nul | findstr /i "torch rasterio numpy" >> logs_env.txt 2>&1
python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.version.cuda)" >> logs_env.txt 2>&1
echo ---- gpu >> logs_env.txt
nvidia-smi >> logs_env.txt 2>&1
echo ---- disk >> logs_env.txt
wmic logicaldisk get caption,freespace,size >> logs_env.txt 2>&1
echo ---- internet >> logs_env.txt
curl -s -o nul -w "%%{http_code}" "https://storage.googleapis.com/storage/v1/b/sen1floods11/o?maxResults=1" >> logs_env.txt 2>&1
echo. >> logs_env.txt
echo DONE >> logs_env.txt
