import os
import tqdm
import numpy
import torch
from typing import Any
from .lrc_encoder import lrc_encoder
from torch.utils.data import Dataset
from .args_type import TrainingArgs


feature_list = [
    ["f0", "wav.pth"],
    ["face", "mp4.pth"],
    ["hubert", "wav.npy"],
    ["text", "wav.pth"],
]


class RWKVTuberDataset(Dataset):
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

                self.sample_list.append(self.enconv_sample(sample))
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
        return self.enconv_sample(sample)

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
    
    def enconv_sample(self, sample):
        if sample[2].shape[0] < 2050:  # pad
            #sample[0] = torch.nn.functional.pad(sample[0], [0, 0, 0, 4101])
            #sample[1] = torch.nn.functional.pad(sample[1], [0, 0, 0, 1025])
            sample[2] = torch.nn.functional.pad(sample[2], [0, 0, 0, 2050])
        #sample[3] = lrc_encoder(sample[3])
        #sample[0] = sample[0][:4100, ::].reshape(1025, 4 * 360)
        #sample[1] = sample[1][:1025, 1::]
        sample[2] = sample[2][:2050, ::].reshape(1025, 2 * 768)
        #sample[3] = sample[3][:1025]
        return sample[2][:-1], sample[2][1:]
        return [torch.cat(sample[:-1], dim = -1), sample[3]]        # f0 f0 f0 f0 face hubert hubert
    
    def deconv_sample(self, sample):
        
        return sample



if __name__ == "__main__":
    
    print(RWKVTuberDataset()[0])
    #for idx, s in enumerate(tqdm.tqdm(RWKVTuberDataset())):
    #    print(idx, s, s.shape)
