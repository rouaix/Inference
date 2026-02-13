@echo off
REM Lance l'interface Gradio P2P Inference
cd /d "%~dp0"

REM Activation du venv
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
    echo [OK] venv active
) else (
    echo [WARN] Pas de venv trouve, utilisation du Python systeme
)

REM Verification et installation de gradio si absent
python -c "import gradio" 2>nul || (
    echo Installation de gradio...
    pip install "gradio>=4.0"
)

REM Lancement de l'interface
echo.
echo  P2P Inference UI - http://127.0.0.1:7860
echo.
python app.py %*
pause
