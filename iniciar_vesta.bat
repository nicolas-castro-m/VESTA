@echo off
REM VESTA - inicio en Windows
cd /d %~dp0
if not exist .venv (
  echo Creando entorno virtual...
  py -3 -m venv .venv
  call .venv\Scripts\activate.bat
  python -m pip install --upgrade pip
  pip install -r requirements.txt
) else (
  call .venv\Scripts\activate.bat
)
streamlit run app.py
