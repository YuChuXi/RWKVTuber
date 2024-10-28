import os
import tqdm
import numpy
import torch
from typing import Any

feature_list = [
    ["f0", "wav.npy"],
    ["f0nsf", "wav.npy"],
    ["face", "mp4.pth"],
    ["hubert", "wav.npy"],
    ["text", "wav.pth"],
]


class RWKVTuberDataset(torch.utils.data.Dataset):
    def __init__(self, dataset="samples", all_in_mem=True):
        self.all_in_mem = all_in_mem
        self.sample_list = []
        self.dataset = dataset

        for idx in tqdm.tqdm(
            os.listdir(f"dataset/{self.dataset}/video"), desc="test|load data"
        ):
            if not self.check_sample(idx[:-4]):
                print(f"!!! skip:{idx}")
                continue
            if self.all_in_mem:
                sample = self.load_sample(idx[:-4])
                if sample[2].shape[0] < 1024: # pad
                    sample[0] = torch.nn.functional.pad(sample[0][:4096], [0, 4096])
                    sample[1] = torch.nn.functional.pad(sample[1][:4096], [0, 4096])
                    sample[2] = torch.nn.functional.pad(
                        sample[2][:1024, 1::], [0, 0, 0, 1024]
                    )
                    sample[3] = torch.nn.functional.pad(
                        sample[3][:2048, ::], [0, 0, 0, 2048]
                    )
                sample[0] = sample[0][:4096].reshape(1024, 4)
                sample[1] = sample[1][:4096].reshape(1024, 4)
                sample[2] = sample[2][:1024, 1::]
                sample[3] = sample[3][:2048, ::].reshape(1024, 2, 768)
                self.sample_list.append(sample)
            else:
                self.sample_list.append(idx[:-4])
            """
            for feature in sample:
                print(idx, feature, feature.shape if hasattr(feature, "shape") else len(feature))
            """

    def __getitem__(self, index) -> Any:
        if self.all_in_mem:
            return self.sample_list[index]
        sample = self.load_sample(self.sample_list[index])
        if sample[2].shape[0] < 1024: # pad
            sample[0] = torch.nn.functional.pad(sample[0][:4096], [0, 4096])
            sample[1] = torch.nn.functional.pad(sample[1][:4096], [0, 4096])
            sample[2] = torch.nn.functional.pad(
                sample[2][:1024, 1::], [0, 0, 0, 1024]
            )
            sample[3] = torch.nn.functional.pad(
                sample[3][:2048, ::], [0, 0, 0, 2048]
            )
        sample[0] = sample[0][:4096].reshape(1024, 4)
        sample[1] = sample[1][:4096].reshape(1024, 4)
        sample[2] = sample[2][:1024, 1::]
        sample[3] = sample[3][:2048, ::].reshape(1024, 2, -1)
        return sample

    def __len__(self):
        return len(self.sample_list)

    def check_sample(self, idx):
        for f in feature_list:
            if os.path.isfile(f"dataset/{self.dataset}/{f[0]}/{idx}.{f[1]}"):
                continue
            return False
        return True

    def load_sample(self, idx):
        features = []
        for f in feature_list:
            file = f"dataset/{self.dataset}/{f[0]}/{idx}.{f[1]}"
            if "npy" in f[1]:
                feature = torch.tensor(numpy.load(file))
            elif "pth" in f[1]:
                feature = torch.load(file, weights_only=True)
            features.append(feature)
        return features

if __name__ == "__main__":
    for idx, s in enumerate(tqdm.tqdm(RWKVTuberDataset())):
        print(idx, s)