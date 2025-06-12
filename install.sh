#!/bin/bash
set -e

echo "==> Aktualizacja pakietów"
sudo apt update

echo "==> Instalacja pakietów systemowych do wxPython"
sudo apt install -y \
  build-essential \
  libgtk-3-dev \
  libgl1-mesa-dev \
  libglu1-mesa-dev \
  libjpeg-dev \
  libtiff-dev \
  libsdl2-dev \
  libnotify-dev \
  libwebkit2gtk-4.0-dev \
  libgstreamer-plugins-base1.0-dev \
  libsm-dev \
  freeglut3-dev \
  libexpat1-dev \
  libcurl4-openssl-dev \
  python3-dev \
  python3-venv \
  python3-serial \
  python3-opengl \
  python3-scipy \
  python3-matplotlib \

echo "==> Tworzenie środowiska wirtualnego"
python3 -m venv venv
source venv/bin/activate

echo "==> Aktualizacja pip"
pip install --upgrade pip

echo "==> Instalacja zależności"
pip install -r requirements.txt

echo "==> Gotowe! Aby aktywować środowisko:"
echo "source venv/bin/activate"

