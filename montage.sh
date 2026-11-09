#!/bin/sh
# montage launcher:  ./montage.sh serve   |   ./montage.sh ask "QUERY"   (needs gglib running with the vision model)
exec uv run -q --with numpy --with torch --with open_clip_torch --with pillow --with mlx-whisper python -I "$(dirname "$0")/mt.py" "$@"
