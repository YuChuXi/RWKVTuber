
mkdir "dataset/samples/25fps_video/" "dataset/samples/video/" "dataset/samples/audio/"
for i in $(ls dataset/samples/raw_video)
do
    #ffmpeg -y -i "dataset/samples/raw_video/$i" -filter:v fps=25 "dataset/samples/25fps_video/$i"
    #ffmpeg -y -i "dataset/samples/raw_video/$i" -c:a pcm_s16le -ar 16000 -ac 1 "dataset/samples/25fps_video/${i/mp4/wav}"
    python video/audio_slicer.py "dataset/samples/25fps_video/${i/mp4/wav}" "dataset/samples/audio"
    python video/video_slicer.py "dataset/samples/25fps_video/$i" "dataset/samples/video"
done