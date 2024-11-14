import sys
import os

os.environ["WKV"] = "fla"
os.environ["RWKV_FLOAT_MODE"] = "bf16"
os.environ["RWKV_VERSION"] = "x060"
os.environ["RWKV_HEAD_SIZE_A"] = "64"
os.environ["RWKV_JIT_ON"] = "true"
os.environ["RWKV_TRAIN_TYPE"] = "none"

import torch
import tqdm
import copy
from .llm_model import RWKV
from .args_type import TrainingArgs
from json2binidx_tool.tools.rwkv_tokenizer import RWKV_TOKENIZER


def __nop(ob):
    return ob

MyModule = torch.nn.Module
MyFunction = __nop
if os.environ["RWKV_JIT_ON"] == "1":
    MyModule = torch.jit.ScriptModule
    MyFunction = torch.jit.script_method

sample = [
    [" President", (0.18, 0.64)],
    [" Biden", (0.64, 1.0)],
    [" will", (1.0, 1.18)],
    [" be", (1.18, 1.3)],
    [" appearing", (1.3, 1.66)],
    [" on", (1.66, 1.98)],
    [" the", (1.98, 2.18)],
    [" right.", (2.18, 2.74)],
    [" A", (2.74, 2.74)],
    [" coin", (2.74, 3.06)],
    [" toss", (3.06, 3.44)],
    [" determined", (3.44, 4.32)],
    [" their", (4.32, 4.58)],
    [" positions.", (4.58, 5.32)],
    [" Each", (5.32, 5.92)],
    [" candidate", (6.06, 6.32)],
    [" will", (6.32, 6.58)],
    [" have", (6.58, 6.74)],
    [" two", (6.74, 7.06)],
    [" minutes", (7.06, 7.38)],
    [" to", (7.38, 7.58)],
    [" answer", (7.58, 7.88)],
    [" a", (7.88, 8.04)],
    [" question", (8.04, 8.44)],
    [" and", (8.44, 9.1)],
    [" one", (9.1, 9.42)],
    [" minute", (9.42, 9.76)],
    [" each", (9.76, 10.0)],
    [" for", (10.0, 10.32)],
    [" responses", (10.32, 10.92)],
    [" and", (11.28, 11.4)],
    [" rebuttals.", (11.4, 12.14)],
    [" An", (12.14, 12.6)],
    [" additional", (12.6, 13.04)],
    [" minute", (13.04, 13.3)],
    [" for", (13.3, 13.54)],
    [" follow", (13.54, 13.92)],
    ["-up", (13.92, 14.1)],
    [" clarification", (14.1, 14.68)],
    [" or", (14.68, 15.12)],
    [" response", (15.12, 15.62)],
    [" is", (15.62, 16.22)],
    [" at", (16.22, 16.42)],
    [" the", (16.42, 16.56)],
    [" moderator's", (16.56, 17.28)],
    [" discretion.", (17.62, 18.22)],
    [" When", (18.22, 18.96)],
    [" it's", (18.96, 19.14)],
    [" time", (19.14, 19.38)],
    [" for", (19.38, 19.6)],
    [" a", (19.6, 19.74)],
    [" candidate", (19.74, 20.08)],
    [" to", (20.08, 20.38)],
    [" speak,", (20.38, 20.84)],
    [" his", (20.84, 21.18)],
    [" microphone", (21.18, 21.68)],
    [" will", (21.68, 22.06)],
    [" be", (22.06, 22.24)],
    [" turned", (22.24, 22.52)],
    [" on", (22.52, 22.94)],
    [" and", (23.24, 23.42)],
    [" his", (23.42, 23.56)],
    [" opponent's", (23.56, 24.04)],
    [" microphone", (24.04, 24.48)],
    [" will", (24.48, 24.9)],
    [" be", (24.9, 25.06)],
    [" turned", (25.06, 25.34)],
    [" off.", (25.32, 26.08)],
    [" Should", (26.08, 26.54)],
    [" a", (26.54, 26.66)],
    [" candidate", (26.66, 27.02)],
    [" interrupt", (27.02, 27.6)],
    [" when", (27.6, 27.98)],
    [" his", (27.98, 28.16)],
    [" microphone", (28.16, 28.6)],
    [" is", (28.6, 28.88)],
    [" muted,", (28.88, 29.38)],
    [" he", (29.38, 29.72)],
    [" will", (29.72, 29.86)],
    [" be", (29.86, 30.04)],
    [" difficult", (30.04, 30.48)],
    [" to", (30.48, 30.76)],
    [" understand", (30.76, 31.34)],
    [" for", (31.34, 31.74)],
    [" viewers", (31.74, 32.22)],
    [" at", (32.22, 32.4)],
    [" home.", (32.4, 33.18)],
    [" At", (33.18, 33.58)],
    [" the", (33.58, 33.68)],
    [" end", (33.68, 33.84)],
    [" of", (33.84, 33.94)],
    [" the", (33.94, 34.04)],
    [" debate,", (34.04, 34.48)],
    [" each", (34.48, 34.66)],
    [" candidate", (34.66, 35.06)],
    [" will", (35.06, 35.36)],
    [" get", (35.36, 35.58)],
    [" two", (35.58, 35.84)],
    [" minutes", (35.84, 36.2)],
    [" for", (36.2, 36.54)],
    [" closing", (36.54, 36.94)],
    [" statements.", (36.94, 37.74)],
    [" There", (37.74, 38.14)],
    [" is", (38.14, 38.32)],
    [" no", (38.32, 38.58)],
    [" studio", (38.58, 39.02)],
    [" audience", (39.02, 39.54)],
    [" tonight.", (39.54, 40.3)],
    [" Pre", (40.3, 40.62)],
    ["-written", (40.62, 40.88)],
]

class XHooker(MyModule):
    def __init__(self, model: RWKV):
        super().__init__()
        self.layer_input = []  # L*(B, T, C)
        self.layer_hooks = []
        self.rwkv_model: RWKV = model
    def start_hook(self):
        self.rwkv_model.eval().cuda()
        for nn in self.rwkv_model.blocks:
            self.layer_hooks.append(nn.register_forward_hook(self.hooker))

    @MyFunction
    def hooker(self, model, input: torch.Tensor, output):
        self.layer_input.append(input[0].cpu())

    def clean_input(self):
        self.layer_input.clear()
        torch.cuda.empty_cache()

    @MyFunction
    def jit1(self, x):
        self.rwkv_model(torch.tensor(x, device="cuda"))
        L = len(self.layer_input)
        B = self.layer_input[0].shape[0]
        T = self.layer_input[0].shape[1]
        C = self.layer_input[0].shape[2]
        inputs = (
            torch.cat(self.layer_input, dim=0)  # (L * B, T, C)
            .reshape(
                L,
                B,
                T,
                C,
            )  # (L, B, T, C)
            .permute(1, 2, 0, 3)  # (B, T, L, C)
            .reshape(
                B,
                T,
                L * C,
            )  # (B, T, L * C)
        )
        return inputs

    def forward(self, x):
        inputs = self.jit1(x)
        self.clean_input()
        return inputs


args = TrainingArgs(
    load_model="RWKVTuber/model/RWKV-x060-World-3B-v2.1-20240417-ctx4096.pth",
    vocab_size=65536,
    ctx_len=512,
    n_embd=2560,
    n_layer=32,
    dim_att=2560,
    dim_ffn=8960,
    fla=True,
)
rwkv = XHooker(RWKV(args))
rwkv.start_hook()

tokenizer = RWKV_TOKENIZER("RWKVTuber/json2binidx_tool/rwkv_vocab_v20230424.txt")
# llm = RWKV()


def lrcs_encoder(lrcs):
    B = len(lrcs)
    idxss = []
    for b in range(B):
        lrc = lrcs[b]
        idxs = []
        for i in range(len(lrc)):
            lrci = lrc[i]
            lrci[0] = tokenizer.encode(lrci[0])
            idxs += lrci[0]
        idxss.append(idxs)

    xs = rwkv(idxss)  # x(B, T) -> (B, T, L, C)
    frame = torch.zeros((B, 1030, xs.shape[-1]))
    xsppd = [0] * B
    for b in range(B):
        for i in range(len(lrc)):
            lrci = lrc[i]
            start = round(lrci[1][0] * 25)
            for j in range(len(lrci[0])):
                frame[b, start + j] = xs[b, xsppd[b]]
                # print(start + j, frame[b, start + j], frame[b, start + j].shape)
                xsppd[b] += 1

    return frame


def lrc_encoder(lrc):
    return lrcs_encoder([lrc])[0]


if __name__ == "__main__":
    print()
    B = 4
    for i in tqdm.trange(512 // B):
        lrcs_encoder([copy.deepcopy(sample) for i in range(B)])
    for i in tqdm.trange(512):
        lrc_encoder(copy.deepcopy(sample))
