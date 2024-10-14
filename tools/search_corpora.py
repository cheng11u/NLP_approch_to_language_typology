import os
from grewpy import Corpus

def select_by_language(annot_scheme: str, dir: str, lang: str, nb_sentences_min: int = 0) -> list[str]:
    """
    Gives the paths to the corpora of a given language.

    Parameters
    ----------
    annot_scheme (str): Annotation scheme (UD or SUD)
    dir (str): Directory containing all the corpora
    lang (str): Language of the corpora
    nb_sentences_min (int): Minimum number of sentences

    Returns
    -------
    list_paths (list[str]): List to the paths of the corpora 
    """
    list_paths = []
    lang_format = "_".join([word.capitalize() for word in lang.split(" ")])
    beginning = "{}_{}-".format(annot_scheme.upper(), lang_format)
    
    # Check all the folders containing the corpora
    for elt in os.listdir(dir):
        if elt.startswith(beginning):
            path = os.path.join(dir, elt)
            if os.path.isdir(path):
                # Check that the corpus has enough sentences
                if nb_sentences_min <= 0:
                    list_paths.append(path)
                else:
                    corpus = Corpus(path)
                    nb_sents = len(corpus.get_sent_ids())
                    if nb_sents >= nb_sentences_min:
                        list_paths.append(path)

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