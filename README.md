
# A NLP approach to language typology
# Research project @ M2 TAL - IDMC

## Group members

- Mehsen Azizi
- Luc Cheng
- Rémi De Vergnette
- Tiankai Luo

## Description

The goal of the project is to compare the distribution of some syntactic patterns in annotated treebanks and between languages. 

To do so, we use several tools, such as Grew and grewpy (see installation section). 

We use the UD 2.14 treebanks for our experiments. 
These treebanks can be downloaded [here](https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-5502#show-files). They should be placed in the data/ud-treebanks/ folder : to download the latest (`udv2.15`) annotations you can run the commands :
```bash
curl --remote-name-all https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-5787{/ud-treebanks-v2.15.tgz,/ud-documentation-v2.15.tgz,/ud-tools-v2.15.tgz}
tar -xvzf ud-treebanks-v2.15.tgz 
mv ud-treebanks-v2.15 ud-treebanks
```


## Installation

The project relies on Grew and the GrewPy library in order to parse and query tree banks.The instructions for installation are given [here](https://grew.fr/usage/install/). These instructions are available for Linux and Mac OS X. If you are using Windows, you can use WSL.

After installing Grew, run the following command:

```pip install -r requirements.txt```

## Running the adj-noun order classifier

### Data

The UD treebanks are expected to be found in the `data/ud-treebanks-v2.14` folder.
#### `an_expe.an_examples_extraction` script
In order to extract adjective-noun pairs from a collection of corpora, use the `an_expe.an_examples_extraction` script. It expects a list of corpora names as arguments, and outputs in the standard output the list of adjective-noun pairs in json format.

Example use 1: to form a adjective noun pairs corpus from the UD_German-LIT corpus and save them in the data folder, run :```python -m an_expe.an_examples_extraction UD_German-LIT>data/an_lit_german.json ``` 

Example use 2 : to form a adjective noun pairs corpus from all the Italian corpora (except the `UD_Italian-Old` one) and save them in the `data/an_it.json` file, run :

```bash
 ls data/ud-treebanks/|grep Italian|grep -v 'UD_Italian-Old'|xargs python -m an_expe.an_examples_extraction>data/an_it.json 
 ``` 

#### `an_expe.an_conditional_examples_extraction` script
This script creates two dataset of adjective noun pairs from a list of corpora, filtering on cosine similarity between adjectives and a set of reference words. 
It works similarly as `an_expe.an_examples_extraction` but expects additional argument : a threshold for cosine similarity, some filter words separated by spaces, and an output path for the dataset with positive example, and the one with negative examples.

Example use 1 : to form adjective noun pairs corpus from French FQB, filtering color adjectives :
```bash
python -m an_expe.an_conditional_examples_extraction UD_French-FQB --threshold 0.5 --filter-words rouge bleu vert --output-positives data/an_colors.json --output-non-positives data/an_uncolored.json
```

Example use 2 : to form adjective noun pairs corpus from all the french corpora, excluding `Old-French`, excluding `UD_Maghrebi_Arabic_French-Arabizi`,`
UD_Middle_French-PROFITEROLE`,`
UD_Old_French-PROFITEROLE`, and filtering on proximity with medical words, you can make use of xargs to run :

```bash
ls data/ud-treebanks/|grep French|grep -v -e 'Maghrebi' -e 'Old' -e 'Middle' -e 'Stories' |xargs echo|xargs -I {} echo {} "--threshold 0.4 --filter-words docteur infirmier chirurgie hopital --output-positives data/an_medical.json --output-non-positives data/an_non_medical.json"|xargs -d " " python -m an_expe.an_conditional_examples_extraction
```


### Training
The `an_expe.train` script trains the adjective noun order classifier, using a json configuration file. If not provided it will use the one in `config/an_expe/config.json`. The optional argument `--ablation`  can be used in order to also train the model using only the adjective, and then only the noun embeddings. 

Here we provide as an example the default configuration file
```json
{

    "seed": 1, # Random seed
    "split": [0.72, 0.18, 0.1], # Train, Test, validation split
    "dataset_path": "data/an.json", #Path of the dataset
    "embeddingModel": "models/cc.fr.300.bin", #Fasttext embedding model
    "num_layers": 4, # number of hidden layers
    "hidden_dim": 100, # Hidden dimension of the encoder
    "dropout": 0.2, # Dropout probability 
    "mu_reg": 1, # (not used anymore) mu for regularisation
    "lambda_reg": 0, # lambda for regularisation (not used anymore)
    "learning_rate": 0.02, # Learning rate for the Adam optimizer
    "batch_size": 16, # Batch size
    "positive_weight": 0.1, # Weight of positive samples (noun before adjective) inside the BCE loss, in order to handle strong imbalances in datasets
    "num_epochs": 12 # Number of training epochs
}
 ```
You will need to use a pretrained fattext word embedding model, that you can download [here](https://fasttext.cc/docs/en/crawl-vectors.html). 

We train with early stopping, the best checkpoint is saved under models/best.pt

Example use of the training script :
`python -m an_expe.train --config data/config/an_expe/config_gold.json`

### Testing  
The training script reports test metrics on an unseen subset of the dataset of size specified by teh split filed of the configuration file. Testing on a different dataset can be doing using the `an_expe.test` script. Like training, it requires a configuration file in order to decide what architecture to use, and will use the weights saved in the `models/best.pt` file. The file under the `dataset_path` will be used for testing the model: this means a given model must be tested using the same configuration file it was trained with, simply changing the path of the dataset used for training.

Example use of the testing script :
`python -m an_expe.train --config data/config/an_expe/config_gold_fqb.json`
to test only on the french FQB tree bank.


## Statistical approach

We carried out an experiment to check whether being in a question changes the 
adjective-noun distribution. To run it use the following command:

```python -m statistical_approach.question_distributions```

