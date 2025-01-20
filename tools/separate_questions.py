
from abc import abstractmethod, ABC
from typing import Union
from conllu import parse

class Separator(ABC):
    """
    Abstract class to separate a corpus according to a specific criterion.
    The criterion is defined in subclasses.
    """
    def __init__(self, data: Union[str, list]):
        super().__init__()
        # If the data is a string, it represents the CoNLL-U data
        if isinstance(data, str):
            self.full_conllu = data
        # If it is a list, then the list is a list of CoNLL-U files
        elif isinstance(data, list):
            self.full_conllu = ""
            for filename in data:
                with open(filename, 'r') as f:
                    content = f.read()
                    self.full_conllu += content + "\n"
        else:
            raise ValueError('Should be a string or a list')
        self.conllu_pos = ""
        self.conllu_neg = ""
        
    @abstractmethod
    def separate(self):
        """
        Separates the CoNLL-U data according to a given criterion.
        The sentences corresponding to the criterion are stored into
        self.conllu_pos whereas other sentences are stored into
        self.conllu_neg
        """
        pass

    def save(self, file_pos: str, file_neg: str):
        """
        Saves the data into two different files. Separates the data
        using the ```separate``` method before saving.

        Parameters
        ----------
        file_pos (str): Path to the file containing the sentences 
        corresponding to the criteria
        file_neg (str): Path to the file containing the sentences
        which do not meet the criteria
        """
        # Separate the data
        self.separate()
        # Save into the files
        with open(file_pos, 'w+') as f_pos:
            f_pos.write(self.conllu_pos)
        with open(file_neg, 'w+') as f_neg:
            f_neg.write(self.conllu_neg)


class QuestionSeparator(Separator):
    def __init__(self, data: Union[str, list]):
        super().__init__(data)

    def separate(self):
        self.conllu_pos = ""
        self.conllu_neg = ""
        sentences = parse(self.full_conllu)
        for sentence in sentences:
            # For each sentence, get its CoNLL-U format
            sentence_conllu = sentence.serialize()
            
            if 'text_en' in sentence.metadata:
                # If there is an translation in English, check whether it ends with ? or ?...
                text_en = sentence.metadata['text_en']
                if text_en.endswith('?') or text_en.endswith('?...'):
                    self.conllu_pos += sentence_conllu
                else:
                    self.conllu_neg += sentence_conllu
            elif 'text' in sentence.metadata:
                # If there is no translation in English, take the original text
                text = sentence.metadata['text']
                if text.endswith('?') or text.endswith('?...'):
                    self.conllu_pos += sentence_conllu
                else:
                    self.conllu_neg += sentence_conllu

