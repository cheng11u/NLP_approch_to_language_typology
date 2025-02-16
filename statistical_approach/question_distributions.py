from grewpy import Corpus, Request
from tools.ordering import compute_ordered_distributions, extract_ordered_examples
from tools.charts.heatmap import HeatMap
from tools.dissimilarity import CosineSimilarity, TotalVariationDistance, CosineSimilarity
from tools.search_corpora import get_conllu_files, select_corpora
from tools.separate_questions import QuestionSeparator
from scipy.stats import fisher_exact
import grewpy


grewpy.set_config("ud")  # ud or basic

pattern_str = """
pattern {
    N [upos="NOUN"];
    A [upos="ADJ"];
    N -[amod]-> A;
}
"""

def separate_conllu(lang: str, path_pos: str, path_neg: str, annot_scheme: str = 'UD',
                  corpora_path_dir: str = 'data/ud-treebanks/'):
    """
    This function merges all the corpora of a given language, and separates the data into
    two files, one containing interrogative sentences, and the other one containing all
    other sentences.

    Parameters
    ----------
    lang (str): Language of the corpora
    path_pos (str): Path to the file containing interrogative sentences
    path_neg (str): Path to the file containing other sentences
    annot_scheme (str): Annotation scheme (UD or SUD)
    corpora_path_dir (str): Path to the directory containing all the corpora
    """
    corpora_paths = [c for c in select_corpora(annot_scheme, corpora_path_dir, lang, None)]
    conllu_files = get_conllu_files(corpora_paths)
    question_sep = QuestionSeparator(conllu_files)
    question_sep.save(path_pos, path_neg)

def get_counts_matrix(pattern: str, conllu_paths: list[str]):
    """
    Gives a matrix where the rows are the patterns (for instance A << N and N << A),
    the first column represents corpus that matches with the criterion (interrogative
    sentences for example), and the second one represents the corpus that does not match.

    Parameters
    ----------
    pattern (str): Grew pattern
    conllu_paths (list[str]): Files corresponding to the different corpora

    Returns
    -------
    counts_matrix (list[list[int]]): Number of occurences of each pattern for each corpus
    """
    request = Request(pattern_str)

    corpora = [Corpus(path) for path in conllu_paths]



    # 2-dimensional list where the rows represent the corpora
    # and the columns represent the patterns (A << N or N << A)
    # the values are the number of occurences
    counts_matrix = []

    distributions, labels, counts = compute_ordered_distributions(request, corpora)

    an = 'A << N'
    na = 'N << A'

    for i, question in enumerate(conllu_paths):
        values_dict = dict()
        for pattern, number in counts[i].items():
            # For each pattern (A << N and N << A), add the number of occurences
            values_dict[pattern] = number
        # For each pattern, if the pattern is not found, the number of
        # occurences is 0
        if an not in values_dict.keys():
            values_dict[an] = 0
        if na not in values_dict.keys():
            values_dict[na] = 0
        counts_matrix.append([values_dict[an], values_dict[na]])

    return counts_matrix



for lang in ['English', 'French']:
    question_corpus_path = f'question_expe/{lang.lower()}_questions.conllu'
    non_question_corpus_path = f'question_expe/{lang.lower()}_non_questions.conllu'
    conllu_paths = [question_corpus_path, non_question_corpus_path]
    separate_conllu(lang, conllu_paths[0], conllu_paths[1])
    counts_matrix = get_counts_matrix(pattern_str, conllu_paths)
    stat, pval = fisher_exact(counts_matrix)
    print(counts_matrix)
    print(f"Fisher exact test between {conllu_paths[0]} and {conllu_paths[1]}: p = {pval:.4f}")


