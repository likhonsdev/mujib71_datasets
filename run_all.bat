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

echo Step 4: Creating LLM dataset...
python create_llm_dataset.py
if %ERRORLEVEL% NEQ 0 (
    echo Error running create_llm_dataset.py
    pause
    exit /b 1
)
echo.

echo Step 5: Enhancing datasets using LLM API...
python enhance_dataset_with_llm.py --model groq --examples 5

echo Step 5b: Advanced datasets enhancement tool prepared...
python advanced_enhance_dataset.py
if %ERRORLEVEL% NEQ 0 (
    echo Error running enhance_dataset_with_llm.py
    pause
    exit /b 1
)
echo.

echo Step 6: Uploading datasets to Hugging Face...
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
