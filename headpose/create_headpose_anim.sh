python animate_headpose.py

ffmpeg -i 'animation12.mp4' -i obama_processed/test/inputs/audio39.wav -c:v copy -map 0:v:0 -map 1:a:0 -c:a aac -b:a 192k headpose12.mp4

ffmpeg \
  -i /Users/nicknikolov/Downloads/final-project/OPENFACE/20160806_003.mp4 \
  -i animation12.mp4 \
  -filter_complex '[0:v]pad=iw*2:ih[int];[int][1:v]overlay=W/2:0[vid]' \
  -map [vid] \
  -c:v libx264 \
  -crf 23 \
  -preset veryfast \
  output12.mp4



