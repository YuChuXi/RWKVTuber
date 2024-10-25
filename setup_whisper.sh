
cd whisper.cpp
./models/download-ggml-model.sh large-v3
make GGML_HIPBLAS=1 -j
cd ..
./test_whisper.sh