#!/bin/bash
# One-click installer for YOLO Real-Time Face Recognition
# Works on macOS and Linux

set -e

echo "=================================================="
echo "  YOLO Real-Time Face Recognition - Installer"
echo "=================================================="
echo ""

# Check Python
if ! command -v python3 &>/dev/null; then
    echo "[ERROR] Python 3 is not installed. Please install Python 3.10+ first."
    echo "        https://www.python.org/downloads/"
    exit 1
fi

PY_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "[✓] Python $PY_VERSION detected"

# Create virtual environment
echo ""
echo "[*] Creating virtual environment..."
python3 -m venv venv
echo "[✓] Virtual environment created"

# Activate and install
echo ""
echo "[*] Installing dependencies (this may take a few minutes)..."
source venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo "[✓] Dependencies installed"

echo ""
echo "=================================================="
echo "  Installation complete!"
echo "  Run the app with:"
echo ""
echo "    source venv/bin/activate"
echo "    python main.py"
echo "=================================================="
