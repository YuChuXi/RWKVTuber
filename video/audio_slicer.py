from pydub import AudioSegment
import sys, os

l = 41


def silce(filename, save_dir):
    audio_segment = AudioSegment.from_file(filename, format="wav")

    total = int(audio_segment.duration_seconds / l)
    print(audio_segment.duration_seconds, audio_segment.duration_seconds // l)
    for i in range(total):
        split_fullname = f"{save_dir}/{filename.split('/')[-1].split('.')[0]}-{i}.wav"
        print(f"Write split {split_fullname}")
        audio_segment[i * l * 1000 : (i + 1) * l * 1000].export(
            split_fullname, format="wav"
        )
    split_fullname = f"{save_dir}/{filename.split('/')[-1].split('.')[0]}-{total}.wav"
    print(f"Write split {split_fullname}")
    audio_segment[total * l * 1000 :].export(
        split_fullname, format="wav"
    )


if __name__ == "__main__":
    os.makedirs(sys.argv[2], exist_ok=True)
    silce(sys.argv[1], sys.argv[2])
