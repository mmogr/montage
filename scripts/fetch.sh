#!/bin/bash
# fetch.sh VIDEO_ID OUTDIR  -> info.json (heatmap, chapters, top comments) + English auto-captions (json3)
id="$1"; out="$2"
[ -f "$out/$id.info.json" ] && exit 0
uvx yt-dlp --skip-download --no-warnings -q \
  --write-info-json --write-comments \
  --write-auto-subs --write-subs --sub-langs "en,en-US,en-orig" --sub-format json3 \
  --extractor-args "youtube:max_comments=300,300,0,0;comment_sort=top" \
  -o "$out/%(id)s.%(ext)s" "https://www.youtube.com/watch?v=$id" 2> "$out/$id.err"
echo "done $id"
