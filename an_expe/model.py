import torch
import numpy as np


class OrderPredictor(torch.nn.Module):

    def __init__(self,
                 embeddingModel,
                 num_layers,
                 input_dim=300,
                 hidden_dim=100,
                 activation=torch.nn.ReLU,
                 use_cache=True,
                 dropout=0.3,
                 ablation_mask=[1, 1],
                 use_linear=False):
        super().__init__()
        self.embeddingModel = embeddingModel
        self.cache = {}
        self.a_module = Encoder(input_dim,
                                hidden_dim,
                                num_layers,
                                activation,
                                dropout=dropout)
        self.n_module = Encoder(input_dim,
                                hidden_dim,
                                num_layers,
                                activation,
                                dropout=dropout)

        self.n_proj = torch.nn.Sequential(torch.nn.Linear(hidden_dim, 1),
                                          torch.nn.ReLU())
        self.a_proj = torch.nn.Sequential(torch.nn.Linear(hidden_dim, 1),
                                          torch.nn.ReLU())
        self.batch_norm = torch.nn.BatchNorm1d(2)
        if use_linear:
            self.out_proj = torch.nn.Linear(2, 1, bias=False)
        self.cache = {}
        self.use_linear = use_linear
        self.ablation_mask = ablation_mask
        self.apply(self.init_weights)

    def forward(self, lemmas):
        if type(lemmas["A"]) != list:
            adj_embedding = torch.tensor(self.__get_embedding(
                lemmas["A"])).unsqueeze(0)
            noun_embedding = torch.tensor(self.__get_embedding(
                lemmas["N"])).unsqueeze(0)
        else:
            adj_lemmas, noun_lemmas = lemmas["A"], lemmas["N"]
            adj_embedding = torch.tensor(
                np.array([
                    self.__get_embedding(adj_lemma) for adj_lemma in adj_lemmas
                ]))
            noun_embedding = torch.tensor(
                np.array([
                    self.__get_embedding(noun_lemma)
                    for noun_lemma in noun_lemmas
                ]))
        return self.forward_embeddings(adj_embedding, noun_embedding)

    def forward_embeddings(self, adj_embedding, noun_embedding):
        noun_embedding = self.n_module(noun_embedding)
        noun_embedding = self.n_proj(noun_embedding) * self.ablation_mask[0]
        adj_embedding = self.a_module(adj_embedding)
        adj_embedding = self.a_proj(adj_embedding) * self.ablation_mask[1]
        embedding = torch.cat((noun_embedding, adj_embedding), dim=-1)
        embedding = self.batch_norm(embedding)
        if self.use_linear:
            return self.out_proj(embedding)
        else:
            return torch.sum(embedding, dim=-1).unsqueeze(-1)

    def __get_embedding(self, word):
        if word not in self.cache.keys():
            self.cache[word] = self.embeddingModel[word]
        return self.cache[word]

    def init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)


class Encoder(torch.nn.Module):

    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 activation,
                 dropout=0.3):
        super().__init__()
        layers = [torch.nn.Linear(input_size, hidden_size, bias=True)]
        layers.append(activation())
        for _ in range(num_layers):
            layers.append(torch.nn.Linear(hidden_size, hidden_size, bias=True))
            layers.append(torch.nn.BatchNorm1d(hidden_size))
            layers.append(activation())
            layers.append(torch.nn.Dropout(dropout))
        self.enc = torch.nn.Sequential(*layers)

    def forward(self, x):
        x = 100 * x
        return self.enc(x)
