#!/bin/bash
git clone https://github.com/cheng11u/NLP_approch_to_language_typology.git
cd NLP_approch_to_language_typology

# Installing GREW, if you have issues with this part, refer to https://grew.fr/usage/install/
opam remote add grew "https://opam.grew.fr"
opam install grew


# Installing python3 dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

#Downloading ud-treebanks
mkdir data
cd data
curl --remote-name-all https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-5787{/ud-treebanks-v2.15.tgz,/ud-documentation-v2.15.tgz,/ud-tools-v2.15.tgz}
tar -xvzf ud-treebanks-v2.15.tgz 
mv ud-treebanks-v2.15 ud-treebanks

# Downloading French fasttext embedding model
cd ..
mkdir models
cd models
curl https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.fr.300.bin.gz --output cc.fr.300.bin.gz
gunzip cc.fr.300.bin.gz


# Creates datasets

cd ..
python3 -m an_expe.an_examples_extraction UD_French-PUD>data/an_pud.json 
python3 -m an_expe.an_examples_extraction UD_French-FQB>data/an_fqb.json 

# Create configuration files
cat << EOF >> config/an_expe/config_fr_pud.json
{

    "seed": 1, 
    "split": [0.7, 0.15, 0.15], 
    "dataset_path": "data/an_pud.json",
    "embeddingModel": "models/cc.fr.300.bin", 
    "num_layers": 4, 
    "hidden_dim": 100, 
    "dropout": 0.2, 
    "mu_reg": 1, 
    "lambda_reg": 0, 
    "learning_rate": 0.02, 
    "batch_size": 16, 
    "positive_weight": 0.1, 
    "num_epochs": 12 
}
EOF



cat << EOF >> config/an_expe/config_fr_fqb.json
{

    "seed": 1, 
    "split": [0.7, 0.15, 0.15], 
    "dataset_path": "data/an_fqb.json", 
    "embeddingModel": "models/cc.fr.300.bin",
    "num_layers": 4, 
    "hidden_dim": 100, 
    "dropout": 0.2, 
    "mu_reg": 1, 
    "lambda_reg": 0, 
    "learning_rate": 0.02, 
    "batch_size": 16, 
    "positive_weight": 0.1, 
    "num_epochs": 12 
}
EOF


# Train model
python3 -m an_expe.train --config config/an_expe/config_fr_fqb.json

# Test model
python3 -m an_expe.test --config config/an_expe/config_fr_pud.json
