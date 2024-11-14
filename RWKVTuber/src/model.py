########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from torch.profiler import profile, record_function, ProfilerActivity

import os
import math
import gc
import importlib
import torch

# torch._C._jit_set_profiling_executor(True)
# torch._C._jit_set_profiling_mode(True)
import torch.nn as nn
from torch.nn import functional as F
from lightning_utilities.core.rank_zero import rank_zero_info, rank_zero_only
import lightning as pl
from lightning.pytorch.strategies import DeepSpeedStrategy
from torch.optim import Adam

if (
    importlib.util.find_spec("deepspeed")
    and os.environ.get("USE_DEEPSPEED", "0") == "1"
):
    import deepspeed
    from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

from .infctx_module import BlockStateList
from .block import Block
from .l2warp import L2Wrap
from .args_type import TrainingArgs

try:
    print("RWKV_VERSION", os.environ["RWKV_VERSION"])
except BaseException:
    os.environ["RWKV_VERSION"] = ""

if os.environ.get("RWKV_OPTIM", None) == "adam_mini":
    from adam_mini import Adam_mini


class RWKVTuber(pl.LightningModule):
    def __init__(self, args: TrainingArgs):
        super().__init__()
        self.args = args
        if not hasattr(args, "dim_att"):
            args.dim_att = args.n_embd
        if not hasattr(args, "dim_ffn"):
            args.dim_ffn = args.n_embd * 4
        if not hasattr(args, "tiny_att_layer"):
            args.tiny_att_layer = -1
        if not hasattr(args, "tiny_att_dim"):
            args.tiny_att_dim = -1
        assert args.n_embd % 32 == 0
        assert args.dim_att % 32 == 0
        assert args.dim_ffn % 32 == 0

        self.ecd_f0 = nn.Linear(args.n_f0, args.n_embd, bias=False)
        self.ecd_hubert = nn.Linear(args.n_hubert, args.n_embd, bias=False)
        self.ecd_face = nn.Linear(args.n_face, args.n_embd, bias=False)
        #self.ecd_logit = nn.Linear(args.n_logit, args.n_embd, bias=False)

        self.blocks = nn.ModuleList([Block(args, i) for i in range(args.n_layer)])

        self.ln_out = nn.LayerNorm(args.n_embd)

        self.dcd_f0 = nn.Linear(args.n_embd, args.n_f0, bias=False)
        self.dcd_hubert = nn.Linear(args.n_embd, args.n_hubert, bias=False)
        self.dcd_face = nn.Linear(args.n_embd, args.n_face, bias=False)
        #self.dcd_logit = nn.Linear(args.n_embd, args.n_logit, bias=False)

        if args.dropout > 0:
            self.drop0 = nn.Dropout(p=args.dropout)

    def configure_optimizers(self):
        args = self.args

        lr_decay = set()
        lr_1x = set()
        lr_2x = set()
        lr_3x = set()
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if (("_w1" in n) or ("_w2" in n)) and (args.layerwise_lr > 0):
                lr_1x.add(n)
            elif (("time_mix" in n) or ("time_maa" in n)) and (args.layerwise_lr > 0):
                if args.my_pile_stage == 2:
                    lr_2x.add(n)
                else:
                    lr_1x.add(n)
            elif (("time_decay" in n) or ("time_daaaa" in n)) and (
                args.layerwise_lr > 0
            ):
                if args.my_pile_stage == 2:
                    lr_3x.add(n)
                else:
                    lr_2x.add(n)
            elif ("time_faaaa" in n) and (args.layerwise_lr > 0):
                if args.my_pile_stage == 2:
                    lr_2x.add(n)
                else:
                    lr_1x.add(n)
            elif ("time_first" in n) and (args.layerwise_lr > 0):
                lr_3x.add(n)
            elif (len(p.squeeze().shape) >= 2) and (args.weight_decay > 0):
                lr_decay.add(n)
            else:
                lr_1x.add(n)

        lr_decay = sorted(list(lr_decay))
        lr_1x = sorted(list(lr_1x))
        lr_2x = sorted(list(lr_2x))
        lr_3x = sorted(list(lr_3x))
        # print('decay', lr_decay)
        # print('1x', lr_1x)
        # print('2x', lr_2x)
        # print('3x', lr_3x)
        param_dict = {n: p for n, p in self.named_parameters()}

        if args.layerwise_lr > 0:
            if args.my_pile_stage == 2:
                optim_groups = [
                    {
                        "params": [param_dict[n] for n in lr_1x],
                        "weight_decay": 0.0,
                        "my_lr_scale": 1.0,
                    },
                    {
                        "params": [param_dict[n] for n in lr_2x],
                        "weight_decay": 0.0,
                        "my_lr_scale": 5.0,
                    },  # test: 2e-3 / args.lr_init},
                    {
                        "params": [param_dict[n] for n in lr_3x],
                        "weight_decay": 0.0,
                        "my_lr_scale": 5.0,
                    },  # test: 3e-3 / args.lr_init},
                ]
            else:
                optim_groups = [
                    {
                        "params": [param_dict[n] for n in lr_1x],
                        "weight_decay": 0.0,
                        "my_lr_scale": 1.0,
                    },
                    {
                        "params": [param_dict[n] for n in lr_2x],
                        "weight_decay": 0.0,
                        "my_lr_scale": 2.0,
                    },
                    {
                        "params": [param_dict[n] for n in lr_3x],
                        "weight_decay": 0.0,
                        "my_lr_scale": 3.0,
                    },
                ]
        else:
            optim_groups = [
                {
                    "params": [param_dict[n] for n in lr_1x],
                    "weight_decay": 0.0,
                    "my_lr_scale": 1.0,
                }
            ]

        if args.weight_decay > 0:
            optim_groups += [
                {
                    "params": [param_dict[n] for n in lr_decay],
                    "weight_decay": args.weight_decay,
                    "my_lr_scale": 1.0,
                }
            ]
            if args.optim == "adam_mini":
                return Adam_mini(
                    self,
                    lr=self.args.lr_init,
                    betas=self.args.betas,
                    eps=self.args.adam_eps,
                    weight_decay=0,
                    model_sharding=True,
                    n_feature=args.n_embd,
                    n_head=args.n_embd // 64,
                    lora_r=8,
                )
            if os.environ.get("USE_DEEPSPEED", "0") == "1":
                if self.deepspeed_offload:
                    return DeepSpeedCPUAdam(
                        optim_groups,
                        lr=self.args.lr_init,
                        betas=self.args.betas,
                        eps=self.args.adam_eps,
                        bias_correction=True,
                        adamw_mode=True,
                        amsgrad=False,
                    )
                return FusedAdam(
                    optim_groups,
                    lr=self.args.lr_init,
                    betas=self.args.betas,
                    eps=self.args.adam_eps,
                    bias_correction=True,
                    adam_w_mode=True,
                    amsgrad=False,
                )
            else:
                # use normal adam
                return Adam(
                    optim_groups,
                    lr=self.args.lr_init,
                    betas=self.args.betas,
                    eps=self.args.adam_eps,
                    weight_decay=0,
                    amsgrad=False,
                )
        else:
            if args.optim == "adam_mini":
                return Adam_mini(
                    self,
                    lr=self.args.lr_init,
                    betas=self.args.betas,
                    eps=self.args.adam_eps,
                    weight_decay=0,
                    model_sharding=True,
                    n_feature=args.n_embd,
                    n_head=args.n_embd // 64,
                    lora_r=8,
                )
            if os.environ.get("USE_DEEPSPEED", "0") == "1":
                if self.deepspeed_offload:
                    return DeepSpeedCPUAdam(
                        optim_groups,
                        lr=self.args.lr_init,
                        betas=self.args.betas,
                        eps=self.args.adam_eps,
                        bias_correction=True,
                        adamw_mode=False,
                        weight_decay=0,
                        amsgrad=False,
                    )
                return FusedAdam(
                    optim_groups,
                    lr=self.args.lr_init,
                    betas=self.args.betas,
                    eps=self.args.adam_eps,
                    bias_correction=True,
                    adam_w_mode=False,
                    weight_decay=0,
                    amsgrad=False,
                )
            else:
                return Adam(
                    optim_groups,
                    lr=self.args.lr_init,
                    betas=self.args.betas,
                    eps=self.args.adam_eps,
                    weight_decay=0,
                    amsgrad=False,
                )

    @property
    def deepspeed_offload(self) -> bool:
        strategy = self.trainer.strategy
        if isinstance(strategy, DeepSpeedStrategy):
            cfg = strategy.config["zero_optimization"]
            return cfg.get("offload_optimizer") or cfg.get("offload_param")
        return False

    def forward(self, f0, hubert, face):
        args = self.args
        B, T, _ = hubert.shape
        assert T <= args.ctx_len, "Cannot forward, model ctx_len is exhausted."

        x = (
            self.ecd_f0(f0)
            + self.ecd_hubert(hubert)
            + self.ecd_face(face)
            #+ self.ecd_logit(logit)
        )
        x_emb = x

        if args.dropout > 0:
            x = self.drop0(x)

        for block in self.blocks:
            if args.grad_cp == 1:
                x = torch_checkpoint(block, x, x_emb, use_reentrant=False)
            else:
                x = block(x, x_emb)

        x = self.ln_out(x)

        return (
            self.dcd_f0(x),
            self.dcd_hubert(x),
            self.dcd_face(x),
            #self.dcd_logit(x),
        )

    def training_step(self, batch, batch_idx):
        args = self.args

        f0, f0t, hubert, hubertt, face, facet = batch #, logit, logitt = batch
        # mask = mask.view(-1)
        # sum_mask = torch.sum(mask).item()
        # if sum_mask == 0:
        #     return torch.tensor([0.0], requires_grad=True)

        #f0, hubert, face, logit = self(f0, hubert, face, logit)
        f0, hubert, face = self(f0, hubert, face)
        
        loss = (
            F.cross_entropy(f0.view(-1, f0.size(-1)), torch.argmax(f0t,dim=-1).view(-1))#!!!!
            + F.mse_loss(hubert, hubertt) *20
            + F.mse_loss(face, facet) *20
            #+ F.mse_loss(logit, logitt, reduction="mean")
        )

        return L2Wrap.apply(loss, f0, hubert, face)
        return L2Wrap.apply(loss, f0, hubert, face, logit)

    def training_step_end(self, batch_parts):
        pass

    def generate_init_weight(self):
        print(
            f"""
############################################################################
#
# Init model weight (slow for large models)...
#
############################################################################
"""
        )
        m = {}
        for n in self.state_dict():
            p = self.state_dict()[n]
            shape = p.shape

            gain = 1.0
            scale = 1.0
            if (
                "ln_" in n
                or ".ln" in n
                or "time_" in n
                or "_mask" in n
                or "pos_emb" in n
                or ".mask." in n
            ):
                if "ln_x.weight" in n:
                    layer_scale = (1 + int(n.split(".")[1])) / self.args.n_layer
                    m[n] = (p * 0.0) + (layer_scale**0.7)
                else:
                    m[n] = p
            else:
                if "ecd" in n:
                    scale = -1 * self.args.lr_init
                else:
                    if shape[0] > shape[1]:
                        gain = math.sqrt(shape[0] / shape[1])

                    zero = [
                        ".att.output.",
                        ".ffn.value.",
                        ".ffn.receptance.",
                        ".ffnPre.value.",
                        ".ffnPre.receptance.",
                        "head_q.",
                        ".oo.",
                        ".rr.",
                    ]

                    for kk in zero:
                        if kk in n:
                            scale = 0
                    if "dcd" in n:
                        scale = 0.5
                    if "head_k." in n:
                        scale = 0.1
                    if "head_q." in n:
                        scale = 0

                print(
                    f"{str(shape[0]).ljust(5)} {str(shape[1]).ljust(5)} {str(scale).ljust(4)} {n}"
                )

                if self.args.accelerator.upper() == "GPU":
                    m[n] = torch.empty((shape[0], shape[1]), device="cuda")
                else:
                    m[n] = torch.empty((shape[0], shape[1]))

                if scale == 0:
                    nn.init.zeros_(m[n])
                elif scale < 0:
                    nn.init.uniform_(m[n], a=scale, b=-scale)
                else:
                    nn.init.orthogonal_(m[n], gain=gain * scale)

            m[n] = m[n].cpu()
            if os.environ["RWKV_FLOAT_MODE"] == "fp16":
                m[n] = m[n].half()
            elif os.environ["RWKV_FLOAT_MODE"] == "bf16":
                m[n] = m[n].bfloat16()

            # if n == "emb.weight":
            #     print(m[n])

        gc.collect()
        torch.cuda.empty_cache()
        return m
