@echo off
cd /d "%~dp0"
echo Starting cipher evaluation...
python -m eval.run_eval
echo.
echo Done! Press any key to close.
pause
