@echo off
echo === Sheikh Mujibur Rahman Bangla NLP Dataset Creator ===
echo.

echo Step 1: Running web scraper to collect data...
python scraper.py
if %ERRORLEVEL% NEQ 0 (
    echo Error running scraper.py
    pause
    exit /b 1
)
echo.

echo Step 2: Downloading Sheikh Mujibur Rahman image...
python download_image.py
if %ERRORLEVEL% NEQ 0 (
    echo Error running download_image.py
    pause
    exit /b 1
)
echo.

echo Step 3: Creating HuggingFace Chat dataset...
python create_chat_dataset.py
if %ERRORLEVEL% NEQ 0 (
    echo Error running create_chat_dataset.py
    pause
    exit /b 1
)
echo.

echo Step 4: Uploading datasets to Hugging Face...
python upload_to_huggingface.py
if %ERRORLEVEL% NEQ 0 (
    echo Error running upload_to_huggingface.py
    pause
    exit /b 1
)
echo.

echo === Process Complete! ===
echo Your datasets have been created and uploaded to Hugging Face.
echo Main Dataset: https://huggingface.co/datasets/likhonsheikhdev/mujib71
echo Chat Dataset: https://huggingface.co/datasets/likhonsheikhdev/mujib71-chat
echo.
pause
