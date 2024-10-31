#./setup_whisper.sh
./test_video.sh $1
./test_f0_hubert.sh $1
./test_whisper.sh $1
./test_face_landmarker.sh $1
./test_dataloader.sh