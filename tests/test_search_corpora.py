import unittest
from tools.search_corpora import select_by_language, get_conllu_files
class TestSelectCorpora(unittest.TestCase):

    def setUp(self):
        self.dir = "data/ud-treebanks-v2.14"

    def testSearchNoMinSentences(self):
        expected = ["data/ud-treebanks-v2.14/UD_French-FQB", "data/ud-treebanks-v2.14/UD_French-GSD", 
                    "data/ud-treebanks-v2.14/UD_French-ParisStories",
                    "data/ud-treebanks-v2.14/UD_French-ParTUT", "data/ud-treebanks-v2.14/UD_French-PUD", 
                    "data/ud-treebanks-v2.14/UD_French-Rhapsodie",
                    "data/ud-treebanks-v2.14/UD_French-Sequoia"]
        self.assertEqual(select_by_language("UD", self.dir, "French"), expected)

    def testSearchMinSentences(self):
        expected = ["data/ud-treebanks-v2.14/UD_French-FQB", "data/ud-treebanks-v2.14/UD_French-GSD", 
                    "data/ud-treebanks-v2.14/UD_French-ParisStories", 
                    "data/ud-treebanks-v2.14/UD_French-Rhapsodie",
                    "data/ud-treebanks-v2.14/UD_French-Sequoia"]
        self.assertEqual(select_by_language("UD", self.dir, "French", nb_sentences_min=2000), expected)

    def testSearchNoMinSentencesWithSpace(self):
        expected = ["data/ud-treebanks-v2.14/UD_Ancient_Greek-Perseus", 
                    "data/ud-treebanks-v2.14/UD_Ancient_Greek-PROIEL", 
                    "data/ud-treebanks-v2.14/UD_Ancient_Greek-PTNK"]
        self.assertEqual(select_by_language("UD", self.dir, "Ancient Greek"), expected)

    def testGetConllu(self):
        expected = ["data/ud-treebanks-v2.14/UD_Catalan-AnCora/ca_ancora-ud-dev.conllu",
                    "data/ud-treebanks-v2.14/UD_Catalan-AnCora/ca_ancora-ud-test.conllu",
                    "data/ud-treebanks-v2.14/UD_Catalan-AnCora/ca_ancora-ud-train.conllu",
                    "data/ud-treebanks-v2.14/UD_Croatian-SET/hr_set-ud-dev.conllu",
                    "data/ud-treebanks-v2.14/UD_Croatian-SET/hr_set-ud-test.conllu",
                    "data/ud-treebanks-v2.14/UD_Croatian-SET/hr_set-ud-train.conllu"]
        self.assertEqual(get_conllu_files(["data/ud-treebanks-v2.14/UD_Catalan-AnCora/", "data/ud-treebanks-v2.14/UD_Croatian-SET/"]), expected)

if __name__ == '__main__':
    unittest.main()