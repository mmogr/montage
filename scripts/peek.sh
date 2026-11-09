#!/bin/bash
# peek.sh VIDEO_ID START_SEC END_SEC OUTDIR -> low-res slice + contact sheet (1 frame / 5s)
id=$1; s=$2; e=$3; out=$4; mkdir -p "$out"
uvx yt-dlp -q --no-warnings -f "bv*[height<=360]+ba/b[height<=360]/worst" \
  --download-sections "*$s-$e" --force-keyframes-at-cuts -o "$out/${id}_$s.%(ext)s" \
  "https://www.youtube.com/watch?v=$id" 2>"$out/${id}_$s.err"
f=$(ls "$out/${id}_$s".* | grep -v err | head -1)
ffmpeg -loglevel error -y -i "$f" -vf "fps=1/5,scale=320:-1,tile=6x3" -frames:v 1 "$out/${id}_$s.sheet.jpg"
echo "$out/${id}_$s.sheet.jpg"
