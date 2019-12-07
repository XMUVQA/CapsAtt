__author__ = 'antony'
import config
from data_loader import VQADataLoader
def make_answer_vocab(adic, vocab_size):
    """
    Returns a dictionary that maps words to indices.
    """
    adict = {'': 0}
    nadict = {'': 1000000}
    vid = 1
    for qid in adic.keys():
        answer_obj = adic[qid]
        answer_list = [ans['answer'] for ans in answer_obj]

        for q_ans in answer_list:
            # create dict
            if adict.has_key(q_ans):
                nadict[q_ans] += 1
            else:
                nadict[q_ans] = 1
                adict[q_ans] = vid
                vid += 1

    # debug
    nalist = []
    for k, v in sorted(nadict.items(), key=lambda x: x[1]):
        nalist.append((k, v))

    # remove words that appear less than once
    n_del_ans = 0
    n_valid_ans = 0
    adict_nid = {}
    for i, w in enumerate(nalist[:-vocab_size]):
        del adict[w[0]]
        n_del_ans += w[1]
    for i, w in enumerate(nalist[-vocab_size:]):
        n_valid_ans += w[1]
        adict_nid[w[0]] = i

    return adict_nid


def make_question_vocab(qdic):
    """
    Returns a dictionary that maps words to indices.
    """
    vdict = {'': 0}
    vid = 1
    for qid in qdic.keys():
        # sequence to list
        q_str = qdic[qid]['qstr']
        q_list = VQADataLoader.seq_to_list(q_str)

        # create dict
        for w in q_list:
            if not vdict.has_key(w):
                vdict[w] = vid
                vid += 1
    return vdict

def make_vocab_files():
    """
    Produce the question and answer vocabulary files.
    """
    print 'making question vocab...', config.QUESTION_VOCAB_SPACE
    qdic, _ = VQADataLoader.load_data(config.QUESTION_VOCAB_SPACE)
    question_vocab = make_question_vocab(qdic)
    print 'making answer vocab...', config.ANSWER_VOCAB_SPACE
    _, adic = VQADataLoader.load_data(config.ANSWER_VOCAB_SPACE)
    answer_vocab = make_answer_vocab(adic, config.ANSWER_DIM)
    return question_vocab, answer_vocab