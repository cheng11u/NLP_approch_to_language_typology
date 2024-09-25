from grewpy import Corpus, GRS


def add_implicit_subject(corpus: Corpus) -> Corpus:
    # TAKEN FROM CHOI 2021
    #
    #

    grs = """
    strat main {
        Onf(conj)
        }
    strat main2 {
        Onf(xcomp)
        }
    rule conj {
        pattern {
            V1 [upos=VERB]; V2 [upos=VERB];
            V1 -[conj]-> V2;
            V1 -[nsubj]-> S1;
        }
        without { V2 -[nsubj]-> S2;}
        without { V2 -[isubj]-> S1; }
        commands { add_edge V2 -[isubj]-> S1;}
    }

    rule xcomp {
        pattern {
            V1 [upos=VERB]; V2 [upos=VERB];
            V1 -[xcomp]-> V2;
            V1 -[obj]-> O1;
        }
        without { V2 -[nsubj]-> S2;}
        without { V2 -[isubj]-> S1; }
        commands { add_edge V2 -[isubj]-> O1;}
    }
    """

    grs = """
    strat main {
        Onf(conj)
        }
    rule conj {
        pattern {
            V1 [upos=VERB]; V2 [upos=VERB];
            V1 -[conj]-> V2;
            V1 -[nsubj]-> S1;
        }
        without { V2 -[nsubj]-> S2;}
        without { V2 -[isubj]-> S1; }
        commands { add_edge V2 -[isubj]-> S1;}
    }

    """
    corpus_ = GRS(grs).apply(corpus)
    return corpus_
