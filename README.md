# A NLP approach to language typology
# Research project @ M2 TAL - IDMC

## Group members

- Mehsen Azizi
- Luc Cheng
- Rémi De Vergnette
- Tiankai Luo

## Description

The goal of the project is to compare the distribution of some syntactic patterns in annotated treebanks and between languages. 
To do so, we use several tools, such as Grew and grewpy. We use the UD 2.14 treebanks for our experiments. 
These treebanks can be downloaded [here](https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-5502#show-files).


## Installation

To install Grewpy, you first need to install Grew. The instructions are given [here](https://grew.fr/usage/install/). These instructions are available for Linux and Mac OS X. If you are using Windows, you can use WSL.

After installing Grew, run the following command:

```pip install -r requirements.txt```

## Similarity of SVO and AN pattern on different corpora
We reproduced the experiments described in [Choi et al. (2021)](https://aclanthology.org/2021.ranlp-1.33.pdf). We compared the Subject-Verb-Object (SVO) distribution 
among French corpora and computed the similarity. We also compared the Adjective-Noun (AN) distribution among different languages.


## Running the adj-noun order classifier
You will need to use a pretrained fattext word embedding model [here](https://fasttext.cc/docs/en/crawl-vectors.html). You can specify the path of the model using the 
embeddingModel field in the configuration file. It is by default: models/cc.fr.300.bin"
Then create a dataset of ordered adjective noun pairs with the command: 
``python -m an_expe.an_examples_extraction > data/an.json``
Finally, you can train and test your classifier:
``python -m an_expe.train`` to use the default configuration file
``python -m an_expe.train --config some_config_file`` to use a custom configuration file
We train with early stopping, the best checkpoint is saved under models/best.pt


## Statistical approach

We carried out an experiment to check whether being in a question changes the 
adjective-noun distribution. To run it use the following command:

```python -m statistical_approach.question_distributions```