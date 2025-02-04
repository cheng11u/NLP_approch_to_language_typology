import os
import re
from grewpy import Corpus

def select_corpora(annot_scheme: str, dir: str, lang: str = None, corpus_name = None, nb_sentences_min: int = 0,
                   contains_words_only = True) -> list[str]:
    """
    Gives the paths to the corpora of a given language and a corpus name.

    Parameters
    ----------
    annot_scheme (str): Annotation scheme (UD or SUD)
    dir (str): Directory containing all the corpora
    lang (str): Language of the corpora
    corpus_name (str): Name of the corpus
    nb_sentences_min (int): Minimum number of sentences
    contains_words_only (bool): If True, remove corpora that don't contain words

    Returns
    -------
    list_paths (list[str]): List to the paths of the corpora 
    """
    list_paths = []
    # Build a regex corresponding to the different criteria
    str_regex = r"{}_".format(annot_scheme.upper())
    if lang is None:
        str_regex += r"[A-Za-z_]+-"
    else:
        lang_format = "_".join([word.capitalize() for word in lang.split(" ")]) + '-'
        str_regex += lang_format

    if corpus_name is None:
        str_regex += '.*'
    else:
        str_regex += corpus_name
    str_regex += r'$'

    # Check all the folders containing the corpora
    for elt in os.listdir(dir):
        if re.match(str_regex, elt):
            path_corpus_dir = os.path.join(dir, elt)
            if os.path.isdir(path_corpus_dir):
                # Get files from the corpus
                conllu_files = get_conllu_files([path_corpus_dir])
                # Check whether the sentence contains words
                has_words = contains_words(conllu_files[0])
                if has_words or not contains_words_only:                    
                    # Check that the corpus has enough sentences
                    if nb_sentences_min <= 0:
                        list_paths.append(path_corpus_dir)
                    else:
                        # Count the sentences (empty lines)
                        nb_sents = 0
                        for file in conllu_files:
                            with open(file, 'r') as f:
                                lines = f.readlines()
                                nb_sents += len([l for l in lines if l == '\n'])
                        if nb_sents >= nb_sentences_min:
                            list_paths.append(path_corpus_dir)

    return list_paths

def get_conllu_files(list_dir: list[str]) -> list[str]:
    """
    Returns the list of CoNLL-U files contained in the directories of a given list.
    The function returns a single list containing all the paths to CoNLL-U files.

    Parameters
    ----------
    list_dir (list[str]): List of directories

    Returns
    -------
    list_conllu (list[str]): List of CoNLL-U files
    """
    list_conllu = []
    for dir in list_dir:
        for elt in os.listdir(dir):
            if elt.endswith('.conllu'):
                path = os.path.join(dir, elt)
                list_conllu.append(path)
    return list_conllu

def contains_words(conllu_file: str) -> bool:
    """
    Checks whether a corpus contains words.

    Parameters
    ----------
    conllu_file (str): Path to the CoNLL-U file

    Returns
    -------
    contains_words (bool): True is the corpus contains words, False otherwise
    """
    try:
        with open(conllu_file, 'r') as f:
            line = f.readline()
            while line != '':
                # If the line starts with '# text = ' and a character which is not '_'
                if re.match('\d+\t[^_]', line):
                    return True
                line = f.readline()
        return False
    except IOError:
        print("IO error")