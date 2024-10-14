import torch
import json
import random


class AdjNounDataset(torch.utils.data.Dataset):

    def __init__(self, data_file, corpora=None):
        data_json = json.load(open(data_file))
        if corpora:
            self.data = [
                corpus["samples"] for corpus in data_json["corpora"]
                if corpus["corpus"] in corpora
            ]
        else:
            self.data = [corpus["samples"] for corpus in data_json["corpora"]]
        self.data_an = [
            example for example_list in self.data
            for example in example_list["A << N"]
        ]
        self.data_na = [
            example for example_list in self.data
            for example in example_list["N << A"]
        ]
        self.samples = [(an, 0)
                        for an in self.data_an] + [(na, 1)
                                                   for na in self.data_na]
        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx_item):
        return self.samples[idx_item]
