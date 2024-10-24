#!/usr/bin/bash

for i in $(ls dataset/samples/video/ | grep "mp4\$")
do 
    echo $i
    VIDEO=dataset/samples/video/$i
    AUDIO=dataset/samples/audio/${i/mp4/wav}
    TEXT=dataset/samples/text/${i/mp4/lrc}
    if [ ! -f $AUDIO ]
    ffmpeg -i $VIDEO -ac 1 -ar 16000 $AUDIO -y
    whisper.cpp/main -m whisper.cpp/models/ggml-large-v3.bin \
        -f $AUDIO \
        -olrc -of $TEXT \
        -l auto \
        -mc 128 \
        --print-colors -ml 1
done