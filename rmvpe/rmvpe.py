import os
import sys
import tqdm
import traceback
import ffmpeg

now_dir = os.getcwd()
sys.path.append(now_dir)
import logging
import torch
import numpy as np

logging.getLogger("numba").setLevel(logging.WARNING)


def load_audio(file, sr):
    try:
        # https://github.com/openai/whisper/blob/main/whisper/audio.py#L26
        # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
        file = clean_path(file)  # 防止小白拷路径头尾带了空格和"和回车
        if os.path.exists(file) == False:
            raise RuntimeError(
                "You input a wrong audio path that does not exists, please fix it!"
            )
        out, _ = (
            ffmpeg.input(file, threads=0)
            .output("-", format="f32le", acodec="pcm_f32le", ac=1, ar=sr)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
        )
    except Exception as e:
        traceback.print_exc()
        raise RuntimeError(f"Failed to load audio: {e}")

    return np.frombuffer(out, np.float32).flatten()


def clean_path(path_str):
    return path_str.strip(" ").strip('"').strip("\n").strip('"').strip(" ")



class FeatureInput(object):
    def __init__(self, samplerate=16000, hop_size=160):
        self.fs = samplerate
        self.hop = hop_size

        self.f0_bin = 256
        self.f0_max = 1100.0
        self.f0_min = 50.0
        self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)

    def compute_f0(self, path, decode = True):
        x = load_audio(path, self.fs)
        # p_len = x.shape[0] // self.hop
        if hasattr(self, "model_rmvpe") == False:
            from rmvpe_model import RMVPE

            print("Loading rmvpe model")
            self.model_rmvpe = RMVPE(
                "rmvpe/rmvpe.pt", is_half=True, device="cuda"
            )
        f0 = self.model_rmvpe.infer_from_audio(x, thred=0.03, decode=decode)
        return f0

    def coarse_f0(self, f0):
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - self.f0_mel_min) * (
            self.f0_bin - 2
        ) / (self.f0_mel_max - self.f0_mel_min) + 1

        # use 0 or 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > self.f0_bin - 1] = self.f0_bin - 1
        f0_coarse = np.rint(f0_mel).astype(int)
        assert f0_coarse.max() <= 255 and f0_coarse.min() >= 1, (
            f0_coarse.max(),
            f0_coarse.min(),
        )
        return f0_coarse

    def go(self, paths):
        if len(paths) == 0:
            print("no-f0-todo")
        else:
            for idx, (inp_path, opt_path1) in tqdm.tqdm(enumerate(paths)):
                try:
                    if (
                        os.path.exists(opt_path1 + ".pth") == True
                    ):
                        continue
                    featur_pit = self.compute_f0(inp_path, False)
                    torch.save(
                        featur_pit,
                        opt_path1 + ".pth",
                    )  # ori
                    if idx == 0:
                        print(featur_pit, featur_pit.shape)
                except:
                    print("f0fail-%s-%s-%s" % (idx, inp_path, traceback.format_exc()))


if __name__ == "__main__":
    exp_dir = sys.argv[1]
    print(" ".join(sys.argv))
    featureInput = FeatureInput()
    paths = []
    inp_root = "%s/audio" % (exp_dir)
    opt_root1 = "%s/f0" % (exp_dir)

    os.makedirs(opt_root1, exist_ok=True)
    for name in sorted(list(os.listdir(inp_root))):
        inp_path = "%s/%s" % (inp_root, name)
        if "spec" in inp_path:
            continue
        opt_path1 = "%s/%s" % (opt_root1, name)
        paths.append([inp_path, opt_path1])
    try:
        featureInput.go(paths)
    except:
        print("f0_all_fail-%s" % (traceback.format_exc()))
