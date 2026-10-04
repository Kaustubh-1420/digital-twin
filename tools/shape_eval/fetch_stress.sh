#!/bin/bash
# Stress-test photos (Unsplash License, free use) for the beta-sensitivity check.
set -e
D="$(cd "$(dirname "$0")" && pwd)/stress"
mkdir -p $D && cd $D
get() { curl -sS -L -o "$1" -w "$1: %{http_code} %{size_download}B\n" "https://images.unsplash.com/$2?w=900&fm=jpg&q=85"; }
get heavy_m.jpg   photo-1573878737679-7587b08b1a0f   # unsplash.com/photos/gouUbL7XIaU
get heavy_f.jpg   photo-1573878591960-37c788c55728   # unsplash.com/photos/ZAbdSzJnFtQ
get thin_m.jpg    photo-1714132948674-1e69b46251cc   # unsplash.com/photos/gwWvr3ab44c
get muscle_m.jpg  photo-1579758682665-53a1a614eea6   # unsplash.com/photos/iUzgePOoGko
# backups
get thin_m2.jpg   photo-1714132948646-599350e7d5d6   # unsplash.com/photos/_4CkudcTYZs
get muscle_m2.jpg photo-1650345735660-3ad5857b287a   # unsplash.com/photos/GiXhK7soQdU
