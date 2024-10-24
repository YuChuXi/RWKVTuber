import sys
import cv2
import os


dura_frames = 1025 # 41*1000/(20*2)

def silce(filename, save_dir):

    cap = cv2.VideoCapture(filename)
    video_fps = int(cap.get(cv2.CAP_PROP_FPS))
    assert video_fps == 25
    video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(video_frames,video_frames//dura_frames)
    frame_size = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )

    fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")


    is_success, bgr_im = cap.read()
    frame_index = 1
    
    split_fullname = f"{save_dir}/{filename.split('/')[-1].split('.')[0]}-{frame_index // dura_frames}.mp4"
    print(f"Write split {split_fullname}")
    v = cv2.VideoWriter(split_fullname, fourcc, video_fps, frame_size)
    v.write(bgr_im)
    while True:
        is_success, bgr_im = cap.read()
        if not is_success:
            cap.release()
            v.release()
            break
        if (frame_index % dura_frames) != 0:
            if v.isOpened():
                v.write(bgr_im)
        if (frame_index % dura_frames) == 0:
            if v.isOpened():
                v.write(bgr_im)
                v.release()

                split_fullname = f"{save_dir}/{filename.split('/')[-1].split('.')[0]}-{frame_index // dura_frames}.mp4"
                v = cv2.VideoWriter(split_fullname, fourcc, video_fps, frame_size)
                print(f"Write split {split_fullname}")

        frame_index += 1
    v.release()
    cap.release()


if __name__ == "__main__":
    os.makedirs(sys.argv[2], exist_ok=True)
    silce(sys.argv[1], sys.argv[2])
