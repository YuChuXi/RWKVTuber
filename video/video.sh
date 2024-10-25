
mkdir "dataset/$1/25fps_video/" "dataset/$1/video/" "dataset/$1/audio/"
for i in $(ls dataset/$1/raw_video)
do
    ffmpeg -y -i "dataset/$1/raw_video/$i" -filter:v fps=25 "dataset/$1/25fps_video/$i"
    ffmpeg -y -i "dataset/$1/raw_video/$i" -c:a pcm_s16le -ar 16000 -ac 1 "dataset/$1/25fps_video/${i/mp4/wav}"
    python video/audio_slicer.py "dataset/$1/25fps_video/${i/mp4/wav}" "dataset/$1/audio"
    python video/video_slicer.py "dataset/$1/25fps_video/$i" "dataset/$1/video"
done