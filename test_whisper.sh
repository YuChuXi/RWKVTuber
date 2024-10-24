#!/usr/bin/bash

mkdir dataset/samples/text/
for i in $(ls dataset/samples/audio)
do 
    echo $i
    AUDIO=dataset/samples/audio/${i/mp4/wav}
    TEXT=dataset/samples/text/${i/mp4/}
    if [ ! -f $TEXT ]
    then
        whisper.cpp/main -m whisper.cpp/models/ggml-large-v3.bin \
            -f $AUDIO \
            -olrc -of $TEXT \
            -l auto \
            -mc 128 \
            --print-colors -ml 1
    fi
done