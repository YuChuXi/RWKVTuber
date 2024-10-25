import os
import sys
import traceback
import fairseq
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F


device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

print(" ".join(sys.argv))
model_path = "hubert/hubert_base.pt"

# wave must be 16k, hop_size=320
def readwave(wav_path, normalize=False):
    wav, sr = sf.read(wav_path)
    assert sr == 16000
    feats = torch.from_numpy(wav).float()
    if feats.dim() == 2:  # double channels
        feats = feats.mean(-1)
    assert feats.dim() == 1, feats.dim()
    if normalize:
        with torch.no_grad():
            feats = F.layer_norm(feats, feats.shape)
    feats = feats.view(1, -1)
    return feats


# HuBERT model
print("load model(s) from {}".format(model_path))
# if hubert model is exist
if os.access(model_path, os.F_OK) == False:
    print(
        "Error: Extracting is shut down because %s does not exist, you may download it from https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main"
        % model_path
    )
    exit(0)

models, saved_cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
    [model_path],
    suffix="",
)
model = models[0]
model = model.to(device)
print("move model to %s" % device)
model = model.half()
model.eval()

def go(wavPath, outPath):
    todo = sorted(list(os.listdir(wavPath)))
    n = max(1, len(todo) // 10)  # 最多打印十条
    if len(todo) == 0:
        print("no-feature-todo")
    else:
        print("all-feature-%s" % len(todo))
        for idx, file in enumerate(todo):
            try:
                if file.endswith(".wav"):
                    wav_path = "%s/%s" % (wavPath, file)
                    out_path = "%s/%s" % (outPath, file + ".npy")

                    if os.path.exists(out_path):
                        continue

                    feats = readwave(wav_path, normalize=saved_cfg.task.normalize)
                    padding_mask = torch.BoolTensor(feats.shape).fill_(False)
                    inputs = {
                        "source": feats.half().to(device),
                        "padding_mask": padding_mask.to(device),
                        "output_layer": 12,
                    }
                    with torch.no_grad():
                        logits = model.extract_features(**inputs)
                        feats = logits[0]

                    feats = feats.squeeze(0).float().cpu().numpy()
                    if np.isnan(feats).sum() == 0:
                        np.save(out_path, feats, allow_pickle=False)
                    else:
                        print("%s-contains nan" % file)
                    if idx % n == 0:
                        print("now-%s,all-%s,%s,%s" % (len(todo), idx, file, feats.shape))
            except:
                print(traceback.format_exc())
        print("all-feature-done")

if __name__ == "__main__":
    exp_dir = sys.argv[1]
    print("exp_dir: " + exp_dir)
    wavPath = "%s/audio" % exp_dir
    outPath = "%s/hubert" % exp_dir
    
    os.makedirs(outPath, exist_ok=True)

    go(wavPath, outPath)