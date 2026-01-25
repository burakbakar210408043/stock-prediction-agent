@echo off
echo ==========================================
echo   BORSA AJANI BASLATILIYOR...
echo   Lutfen tarayicinin acilmasini bekle.
echo ==========================================

:: Kodun oldugu klasore git
cd /d "%~dp0"

:: Uygulamayi calistir
py -m streamlit run main.py

:: Eger hata verirse pencere hemen kapanmasin diye bekle
pause
