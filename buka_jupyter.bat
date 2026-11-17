@echo off
title Jupyter Notebook - Proyek Hemat Air
echo ============================================
echo   Membuka Jupyter Notebook: Proyek Hemat Air
echo ============================================
echo.

cd /d "%~dp0"

:: Gunakan Python 3.9 yang memiliki Jupyter
set JUPYTER="C:\Users\Raihan aryanova\AppData\Local\Programs\Python\Python39\Scripts\jupyter.exe"

if exist %JUPYTER% (
    echo [OK] Jupyter ditemukan di Python 3.9
    echo Membuka notebook...
    echo.
    %JUPYTER% notebook analisis_air_wudhu.ipynb
) else (
    echo [COBA] Mencari Jupyter di PATH...
    jupyter notebook analisis_air_wudhu.ipynb
)

pause
