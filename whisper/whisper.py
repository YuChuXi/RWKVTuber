import torch
import sys, os
import tqdm
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3-turbo"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id,
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=True,
    use_safetensors=True,
    cache_dir="whisper/models_cache",
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)


def dlrc(inf, outf, bsz=4):
    infls = set(os.listdir(inf))
    
    bar = tqdm.tqdm(total=len(infls))
    while (linfls := len(infls)) > 0:
        lbsz = min(bsz, linfls)
        input = [infls.pop() for i in range(lbsz)]
        result = pipe(
            [f"{inf}/{i}" for i in input],
            batch_size=lbsz,
            return_timestamps="word",
            chunk_length_s=24,
        )
        bar.update(lbsz)
        for i, o in zip(input, result):
            lt = []
            for t in o["chunks"]:
                lt.append([t["text"], t["timestamp"]])
            torch.save(lt, f"{outf}/{i}.pth")

    print(lt)


if __name__ == "__main__":
    os.makedirs(sys.argv[2], exist_ok=True)
    dlrc(sys.argv[1], sys.argv[2])

