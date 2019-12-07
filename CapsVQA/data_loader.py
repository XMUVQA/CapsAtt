import numpy as np
import re, json, random
import config
import spacy

QID_KEY_SEPARATOR = '/'
GLOVE_EMBEDDING_SIZE = config.GLOVE_DIM


class VQADataLoader(object):

    def __init__(self, folder=config.FOLDER_NAME, batchsize=config.BATCH_SIZE, max_length=15, mode='train'):
        self.batchsize = batchsize
        self.epoch_counter=0
        self.d_vocabulary = None
        self.batch_index = None
        self.batch_len = None
        self.rev_adict = None
        self.max_length = max_length
        self.mode = mode
        if self.mode=='train':
            self.qdic, self.adic = VQADataLoader.load_data(config.TRAIN_DATA_SPLITS)
        else:
            self.qdic, self.adic = VQADataLoader.load_data(self.mode)

        with open('./%s/vdict.json' % folder, 'r') as f:
            self.vdict = json.load(f)
        with open('./%s/adict.json' % folder, 'r') as f:
            self.adict = json.load(f)

        self.n_ans_vocabulary = len(self.adict)
        self.nlp = spacy.load('en_vectors_web_lg')
        self.glove_dict = {}  # word -> glove vector

    @staticmethod
    def load_vqa_json(data_split):
        """
        Parses the question and answer json files for the given data split.
        Returns the question dictionary and the answer dictionary.
        """
        qdic, adic = {}, {}

        with open(config.DATA_PATHS[data_split]['ques_file'], 'r') as f:
            qdata = json.load(f)['questions']
            for q in qdata:
                qdic[data_split + QID_KEY_SEPARATOR + str(q['question_id'])] = \
                    {'qstr': q['question'], 'iid': q['image_id']} # n-iid: q['neg_img_id]

        if 'test' not in data_split:
            with open(config.DATA_PATHS[data_split]['ans_file'], 'r') as f:
                adata = json.load(f)['annotations']
                for a in adata:
                    adic[data_split + QID_KEY_SEPARATOR + str(a['question_id'])] = \
                        a['answers'] # n-answers

        print 'parsed', len(qdic), 'questions for', data_split
        return qdic, adic

    @staticmethod
    def conver_ans2id(adic):
        ans2id={}
        for key,value in adic.items():
            ans2id[value]=key
        return ans2id

    @staticmethod
    def load_genome_json():
        """
        Parses the genome json file. Returns the question dictionary and the
        answer dictionary.
        """
        qdic, adic = {}, {}

        with open(config.DATA_PATHS['genome']['genome_file'], 'r') as f:
            qdata = json.load(f)
            for q in qdata:
                key = 'genome' + QID_KEY_SEPARATOR + str(q['id'])
                qdic[key] = {'qstr': q['question'], 'iid': q['image']}
                adic[key] = [{'answer': q['answer']}]

        print 'parsed', len(qdic), 'questions for genome'
        return qdic, adic

    @staticmethod
    def load_data(data_split_str):
        all_qdic, all_adic = {}, {}
        for data_split in data_split_str.split('+'):
            assert data_split in config.DATA_PATHS.keys(), 'unknown data split'
            if data_split == 'genome':
                qdic, adic = VQADataLoader.load_genome_json()
                all_qdic.update(qdic)
                all_adic.update(adic)
            else:
                qdic, adic = VQADataLoader.load_vqa_json(data_split)
                all_qdic.update(qdic)
                all_adic.update(adic)
        return all_qdic, all_adic

    def getQuesIds(self):
        return self.qdic.keys()

    def getStrippedQuesId(self, qid):
        return qid.split(QID_KEY_SEPARATOR)[1]

    def getImgId(self, qid):
        return self.qdic[qid]['iid']

    def getQuesStr(self, qid):
        return self.qdic[qid]['qstr']

    def getAnsObj(self, qid):
        if self.mode == 'test-dev' or self.mode == 'test':
            return -1
        return self.adic[qid]

    @staticmethod
    def seq_to_list(s):
        t_str = s.lower()
        for i in [r'\?', r'\!', r'\'', r'\"', r'\$', r'\:', r'\@', r'\(', r'\)', r'\,', r'\.', r'\;']:
            t_str = re.sub(i, '', t_str)
        for i in [r'\-', r'\/']:
            t_str = re.sub(i, ' ', t_str)
        q_list = re.sub(r'\?', '', t_str.lower()).split(' ')
        q_list = filter(lambda x: len(x) > 0, q_list)
        return q_list

    def extract_answer(self, answer_obj):
        """ Return the most popular answer in string."""
        if self.mode == 'test-dev' or self.mode == 'test':
            return -1
        answer_list = [answer_obj[i]['answer'] for i in xrange(10)]
        dic = {}
        for ans in answer_list:
            if dic.has_key(ans):
                dic[ans] += 1
            else:
                dic[ans] = 1
        max_key = max((v, k) for (k, v) in dic.items())[1]
        return max_key

    def extract_answer_prob(self, answer_obj):
        """ Return the most popular answer in string."""
        if self.mode == 'test-dev' or self.mode == 'test':
            return -1

        answer_list = [ans['answer'] for ans in answer_obj]
        prob_answer_list = []
        for ans in answer_list:
            if self.adict.has_key(ans):
                prob_answer_list.append(ans)

        if len(prob_answer_list) == 0:
            if self.mode == 'val' or self.mode == 'test-dev' or self.mode == 'test':
                return 'hoge'
            else:
                raise Exception("This should not happen.")
        else:
            return random.choice(prob_answer_list)
    def get_score(self,count):
        if count==0:
            return 0
        elif count==1:
            return 0.3
        elif count==2:
            return 0.6
        elif count==3:
            return 0.9
        else:
            return 1.0
    def extract_answer_list(self, answer_obj):
        answer_list = [ans['answer'] for ans in answer_obj]
        prob_answer_vec = np.zeros(config.ANSWER_DIM)
        answer_count={}
        for ans in answer_list:
            if self.adict.has_key(ans):
                index = self.adict[ans]
                answer_count[index]=answer_count.get(index,0)+1
        for index,count in answer_count.items():
            prob_answer_vec[index]=self.get_score(count)
        #print prob_answer_vec
        return prob_answer_vec

    def qlist_to_vec(self, max_length, q_list):
        """
        Converts a list of words into a format suitable for the embedding layer.

        Arguments:
        max_length -- the maximum length of a question sequence
        q_list -- a list of words which are the tokens in the question

        Returns:
        qvec -- A max_length length vector containing one-hot indices for each word
        cvec -- A max_length length sequence continuation indicator vector
        glove_matrix -- A max_length x GLOVE_EMBEDDING_SIZE matrix containing the glove embedding for
            each word
        """
        glove_matrix = []

        q_len = len(q_list)
        if q_len > self.max_length:
            q_len = self.max_length
        for i in range(max_length):
            if i < q_len:
                w=q_list[i]
                if w not in self.glove_dict:
                    self.glove_dict[w]=self.nlp(u'%s'%w).vector
                glove_matrix.append(self.glove_dict[w])
            else:
                glove_matrix.append(np.zeros(config.GLOVE_DIM,dtype=float))
        return glove_matrix, q_len

    def answer_to_vec(self, ans_str):
        """ Return answer id if the answer is included in vocabulary otherwise '' """
        if self.mode == 'test-dev' or self.mode == 'test':
            return np.zeros(config.ANSWER_DIM,float)

        if self.adict.has_key(ans_str):
            ans = self.adict[ans_str]
        else:
            ans = self.adict['']
        ans_vec=np.zeros(config.ANSWER_DIM,float)
        ans_vec[ans]=1
        return ans_vec

    def vec_to_answer(self, ans_symbol):
        """ Return answer id if the answer is included in vocabulary otherwise '' """
        if self.rev_adict is None:
            rev_adict = {}
            for k, v in self.adict.items():
                rev_adict[v] = k
            self.rev_adict = rev_adict

        return self.rev_adict[ans_symbol]

    def create_batch(self, qid_list):

        ivec = []
        avec = []
        glove_matrix = []
        q_str_list=[]
        q_len_lists=[]

        for i, qid in enumerate(qid_list):

            # load raw question information
            q_str = self.getQuesStr(qid)
            q_ans = self.getAnsObj(qid)
            q_iid = self.getImgId(qid)
            q_str_list.append(q_str)
            # convert question to vec
            q_list = VQADataLoader.seq_to_list(q_str)
            t_glove_matrix,t_q_len= self.qlist_to_vec(self.max_length, q_list)
            q_len_lists.append(t_q_len)

            try:
                qid_split = qid.split(QID_KEY_SEPARATOR)
                data_split = qid_split[0]
                if data_split == 'genome':
                    t_ivec = np.load(config.DATA_PATHS['genome']['features_prefix'] + str(q_iid) + '.jpg.npz')['x']
                else:
                    t_ivec = \
                    np.load(config.DATA_PATHS[data_split]['features_prefix'] + str(q_iid) + '.npy')
                #t_ivec = (t_ivec / np.sqrt((t_ivec ** 2).sum()))
            except:
                t_ivec = 0.
                # n_t_ivec $ negative iamge features
                print 'data not found for qid : ', q_iid, self.mode

            # convert answer to vec
            if self.mode == 'val' or self.mode == 'test-dev' or self.mode == 'test':
                q_ans_str = self.extract_answer(q_ans)
                t_avec = self.answer_to_vec(q_ans_str)
            else:
                t_avec = self.extract_answer_list(q_ans)

            ivec.append(t_ivec)
            avec.append(t_avec)
            glove_matrix.append(t_glove_matrix)

        return q_str_list,glove_matrix,q_len_lists,avec,ivec

    def next_batch(self,batchsize):
        if self.batch_len is None:
            self.n_skipped = 0
            qid_list = self.getQuesIds()
            random.shuffle(qid_list)
            self.qid_list = qid_list
            self.batch_len = len(qid_list)
            self.batch_index = 0
            self.epoch_counter = 0

        def has_at_least_one_valid_answer(t_qid):
            answer_obj = self.getAnsObj(t_qid)
            answer_list = [ans['answer'] for ans in answer_obj]
            for ans in answer_list:
                if self.adict.has_key(ans):
                    return True

        counter = 0
        t_qid_list = []
        t_iid_list = []
        while counter < batchsize:
            t_qid = self.qid_list[self.batch_index]
            t_iid = self.getImgId(t_qid)
            if self.mode == 'val' or self.mode == 'test-dev' or self.mode == 'test':
                t_qid_list.append(t_qid)
                t_iid_list.append(t_iid)
                counter += 1
            elif has_at_least_one_valid_answer(t_qid):
                t_qid_list.append(t_qid)
                t_iid_list.append(t_iid)
                counter += 1
            else:
                self.n_skipped += 1

            if self.batch_index < self.batch_len - 1:
                self.batch_index += 1
            else:
                self.epoch_counter += 1
                qid_list = self.getQuesIds()
                random.shuffle(qid_list)
                self.qid_list = qid_list
                self.batch_index = 0
                print("%d questions were skipped in a single epoch" % self.n_skipped)
                self.n_skipped = 0

        t_batch = self.create_batch(t_qid_list)
        return t_batch + (t_qid_list, t_iid_list,self.epoch_counter) # q_str_list,glove_matrix,q_len_lists,avec,ivec, qid_list, img_id_list