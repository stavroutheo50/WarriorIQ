$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $ProjectPython)) {
    throw "The project .venv is missing. In PyCharm, create a Python 3.10+ virtual environment named .venv, then run this installer again."
}

& $ProjectPython -m pip install --upgrade pip
& $ProjectPython -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
& $ProjectPython -m pip install -r (Join-Path $ProjectRoot "requirements.txt")

& $ProjectPython (Join-Path $ProjectRoot "tools\verify_project.py")
