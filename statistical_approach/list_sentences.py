import grewpy
from tools.ordering import extract_ordered_examples
from grewpy import Corpus, Request

# This program gives the questions in English corpora containing the NA pattern
na = 'N << A'
pattern_str = f'''
pattern {{
    N [upos="NOUN"];
    A [upos="ADJ"];
    N -[amod]-> A;
}}
'''
corpus = Corpus('question_expe/english_questions.conllu')
request = Request(pattern_str)

sentence_ids = list()
examples = extract_ordered_examples(request, [corpus])

# Extract sentence IDs
for pattern in examples:
    list_na = pattern[na]
    for element in list_na:
        sentence_ids.append(element["sent_id"])

# Get the text corresponding to sentence IDs
for sentence in corpus[:len(corpus)]:
    if sentence.meta["sent_id"] in sentence_ids:
        print(sentence.meta["sent_id"], sentence.meta["text"])
        print()