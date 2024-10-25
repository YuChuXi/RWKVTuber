#!/usr/bin/bash

mkdir dataset/samples/text/

python whisper/whisper.py dataset/samples/audio dataset/samples/text
