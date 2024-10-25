#!/usr/bin/bash

mkdir dataset/$1/text/

python whisper/whisper.py dataset/$1/audio dataset/$1/text
