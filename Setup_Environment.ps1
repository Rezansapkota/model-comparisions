$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
& py -3.12 -m venv .venv
if ($LASTEXITCODE -ne 0) { throw 'Python 3.12 is required. Install Python 3.12 and retry.' }
$taskPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
& $taskPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw 'Could not update pip.' }
& $taskPython -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
Write-Host 'Setup complete. Double-click Open_Notebook.cmd to open JupyterLab.'
