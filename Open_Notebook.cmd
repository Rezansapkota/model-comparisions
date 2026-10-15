@echo off
cd /d "%~dp0"
set "TASK_PYTHON=%~dp0.venv\Scripts\python.exe"
if exist "%TASK_PYTHON%" goto launch
echo Run Setup_Environment.ps1 first to install the notebook environment.
pause
exit /b 1
:launch
"%TASK_PYTHON%" -m jupyterlab "Model_Comparison.ipynb"
