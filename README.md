# A NLP approach to language typology
# Research project @ M2 TAL - IDMC

The goal of the project is to compare the distribution of some syntactic patterns in annotated treebanks and between languages. In order to do so, we use several tools, such as Grew and grewpy.

We reproduced the experiments described in [Choi et al. (2021)](https://aclanthology.org/2021.ranlp-1.33.pdf). We compared the Subject-Verb-Object (SVO) distribution among French corpora, and computed the similarity. We also compared the Adjective-Noun (AN) distribution among different languages.

## Installation

Run the following command:

```pip install -r requirements.txt```

## Running the adj-noun order classifier
You will need to use a pretrained fattext word embedding model [here](https://fasttext.cc/docs/en/crawl-vectors.html). You can specify the path of the model using the embeddingModel field in the configuration file. It is by default : models/cc.fr.300.bin"

### Preparing 
``python -m an_expe.an_examples_extraction > data/an.json``
Finally, you can train and test your classifier :
``python -m an_expe.train`` to use the default configuration file
``python -m an_expe.train --config some_config_file`` to use a custom configuration file
We train with early stopping, the best checkpoint is saved under models/best.pt
