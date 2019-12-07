import tensorflow as tf
import copy

import numpy as np

import tensorflow.contrib.slim as slim
import show_utils as utils

MAX_STEPS = 30

def weight_variable(shape,dtype=tf.float32,name=None,lamda=0.0001):
    var=tf.get_variable(name,shape,dtype=dtype,initializer=tf.contrib.layers.xavier_initializer())
    tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(lamda)(var))
    return var


class bottomCapAttV3():
    def __init__(self,opt):
        self.opt = opt
        self.vocab_size = opt.vocab_size
        self.input_encoding_size = opt.input_encoding_size
        self.drop_prob_lm = opt.drop_prob_lm
        self.seq_length  = opt.seq_length
        self.rnn_size = opt.rnn_size
        self.seq_per_img = opt.seq_per_img
        self.batch_size = opt.batch_size
        self.seq_per_img = opt.seq_per_img
        self.att_size = opt.att_size
        self.num_boxes = opt.num_boxes
        
        # Variable indicating in training mode or evaluation mode
        self.training = tf.Variable(True, trainable = False, name = "training")
        self.num_layers = 1
        self.cap_iter = opt.cap_iter

        self.b = tf.placeholder(tf.float32, [None, self.num_boxes, 1, 1])
        if self.opt.cnn_model=="frcnn":
            print "using frcnn feature"
            self.cnn_dim = 2048
            self.images = tf.placeholder(tf.float32,[None,self.num_boxes,self.cnn_dim],name="features")
            self.context = self.images
        elif self.opt.cnn_model == 'vgg16' or self.opt.cnn_model == 'vgg19':
            print "using cnn model"

            self.images = tf.placeholder(tf.float32, [None, 224, 224, 3], name = "images")
            cnn_weight = vars(self.opt).get('cnn_weight', None)
            if self.opt.cnn_model == 'vgg16':
                import vgg
                self.cnn = vgg.Vgg16(cnn_weight)
                self.cnn_dim = 512
            elif self.opt.cnn_model == 'vgg19':
                import vgg
                self.cnn = vgg.Vgg19(cnn_weight)
                self.cnn_dim = 512
            with tf.variable_scope("cnn"):
                self.cnn.build(self.images)

            if self.opt.cnn_model == 'vgg16':
                self.context = self.cnn.conv5_3
            elif self.opt.cnn_model == 'vgg19':
                self.context = self.cnn.conv5_4
            self.context = tf.reshape(self.context, [-1, self.num_boxes, self.cnn_dim])
        elif self.opt.cnn_model == 'resnet':
            print "using resnet feature"
            self.cnn_dim = 2048
            self.images = tf.placeholder(tf.float32,[None,self.num_boxes,self.cnn_dim],name="features")
            self.context = self.images     
            
        self.labels = tf.placeholder(tf.int32, [None, self.seq_length+2], name="labels")
        


        self.masks = tf.placeholder(tf.float32, [None, self.seq_length+2], name="masks")

        

        with tf.variable_scope("rnnlm"):
            #l2_norm
            self.features = tf.nn.l2_normalize(self.context,axis=-1)

            #self.att_feat = slim.fully_connected(self.features, self.att_size,activation_fn=None, scope='att_feature_proj')
            
            self.avgFeat = tf.reduce_mean(self.features,axis=1,keep_dims=False)
            # Word Embedding table
            self.Wemb = tf.Variable(
                tf.random_uniform([self.vocab_size, self.input_encoding_size], -0.1, 0.1), 
                name='Wemb')
            # RNN cell
            
            if opt.rnn_type == 'rnn':
                self.cell_fn = cell_fn = tf.contrib.rnn.BasicRNNCell
            elif opt.rnn_type == 'gru':
                self.cell_fn = cell_fn = tf.contrib.rnn.GRUCell
            elif opt.rnn_type == 'lstm':
                self.cell_fn = cell_fn = tf.contrib.rnn.LSTMCell
            else:
                raise Exception("RNN type not supported: {}".format(opt.rnn_type))

            self.keep_prob = tf.cond(self.training,
                            lambda:tf.constant(1-self.drop_prob_lm),
                            lambda:tf.constant(1.0),name='keep_prob')

            # basic cell has dropout wrapper
            self.basic_cell1 = cell1 = tf.contrib.rnn.DropoutWrapper(cell_fn(self.rnn_size), 
                                        1.0, self.keep_prob)
            self.basic_cell2 = cell2 = tf.contrib.rnn.DropoutWrapper(cell_fn(self.rnn_size), 
                                        1.0, self.keep_prob)
            # cell is the final cell of each timestep
            self.cell1 = tf.contrib.rnn.MultiRNNCell([cell1] * self.num_layers)
            self.cell2 = tf.contrib.rnn.MultiRNNCell([cell2] * self.num_layers)

            
    def initialize(self,sess):
        sess.run(tf.global_variables_initializer())
        # Initialize the saver
        self.saver = tf.train.Saver(tf.trainable_variables())
        if self.opt.load_pre_model:
            self.saver.restore(sess, self.opt.pre_model_path)
        if self.opt.eval_or_train =="train":
            self.summary_writer = tf.summary.FileWriter(self.opt.checkpoint_path, sess.graph)
        print ("total paramter num:%d"%utils.get_num_params())

            
    def get_attention(self,pre_h,pctx,scope="attention",reuse=False):
        with tf.variable_scope(scope,reuse=reuse):
            pstate = slim.fully_connected(pre_h, self.att_size, activation_fn = None, scope = 'h_att') 
            pctx_ = pctx + tf.expand_dims(pstate, 1) 
            pctx_ = tf.nn.tanh(pctx_) 
            alpha = slim.fully_connected(pctx_, 1, activation_fn = None, scope = 'alpha') 
            alpha = tf.squeeze(alpha, [2]) 
            alpha = tf.nn.softmax(alpha)
        return alpha

    def capsule_attention(self, prev_h, flattened_ctx, b_IJ, r_num, iter=3):

        F_reshape = tf.reshape(flattened_ctx, [-1, self.cnn_dim])  # N*R, 512
        W_F = weight_variable([self.cnn_dim, self.att_size], name='imag_proj')
        F_p = tf.matmul(F_reshape, W_F)
        F_p = tf.reshape(F_p, [-1, r_num, self.att_size])
        F_p = tf.expand_dims(F_p, axis=3)  # N, R, A, 1
        F_p_stop = tf.stop_gradient(F_p, name='F_p_stop_gradient')

        W_h = weight_variable([self.rnn_size, self.att_size], name='h_proj')
        h_proj = tf.matmul(prev_h, W_h)
        h_proj = tf.expand_dims(h_proj, 1)  # N,1,A
        h_proj = tf.expand_dims(h_proj, 3)  # N,1, A, 1

        c_IJ_list = []

        with tf.variable_scope('routing'):
            for i in range(iter):
                with tf.variable_scope('iter_' + str(i)):
                    c_IJ = tf.nn.softmax(b_IJ, dim=1)
                    c_IJ_list.append(c_IJ)

                    if i == iter - 1:
                        f_v = tf.multiply(c_IJ, F_p)
                        f_v = tf.reduce_sum(f_v, axis=1, keep_dims=True)  # batch * seq_per_img, 1, h_dim, 1
                        h_proj = h_proj + f_v
                    else:
                        h_proj_tile = tf.tile(h_proj, [1, r_num, 1, 1])
                        u_produce_v = tf.matmul(F_p_stop, h_proj_tile, transpose_a=True)
                        b_IJ += u_produce_v
                        f_v = tf.multiply(c_IJ, F_p_stop)
                        f_v = tf.reduce_mean(f_v, axis=1, keep_dims=True)
                        h_proj = h_proj + f_v

        h_out = tf.reshape(h_proj, [-1, self.att_size])
        alpha = tf.reshape(c_IJ, [-1, self.num_boxes])

        return alpha, h_out, c_IJ_list

               


    def build_model(self):

        with tf.variable_scope("rnnlm"):
            flattened_ctx = self.features

            initial_state1 = utils.get_initial_state(self.avgFeat, self.cell1.state_size,scope="init_state1")
            initial_state2 = utils.get_initial_state(self.avgFeat, self.cell2.state_size,scope="init_state2")

            initial_state1 = utils.expand_feat(initial_state1, self.seq_per_img)
            initial_state2 = utils.expand_feat(initial_state2, self.seq_per_img)

            flattened_ctx = tf.reshape(
                tf.tile(tf.expand_dims(flattened_ctx, 1), [1, self.seq_per_img, 1, 1]),
                [-1, self.num_boxes, self.cnn_dim])
            """
            att_feat_flatten = tf.reshape(
                tf.tile(tf.expand_dims(self.att_feat, 1), [1, self.seq_per_img, 1, 1]),
                [-1, self.num_boxes, self.att_size])
            """
            avg_feat_flatten = tf.reshape(
                tf.tile(tf.expand_dims(self.avgFeat, 1), [1, self.seq_per_img, 1]),
                [-1, self.cnn_dim])

            rnn_inputs = tf.split(axis=1, num_or_size_splits=self.seq_length + 1, 
                            value=tf.nn.embedding_lookup(self.Wemb, 
                                            self.labels[:,:self.seq_length + 1]))# batch_size,1,1000
            rnn_inputs = [tf.squeeze(input_, [1]) for input_ in rnn_inputs]#batch_size,1000

            h0 = utils.last_hidden_vec(initial_state1)
            h1 = utils.last_hidden_vec(initial_state2)
            self.logits = []

            state1 = initial_state1
            state2 = initial_state2
            loss = 0.0
            for t in range(self.seq_length+1): 
                if t > 0:
                    # Reuse the variables after the first timestep.
                    tf.get_variable_scope().reuse_variables()
                x1 = tf.concat([h1,avg_feat_flatten,rnn_inputs[t]],axis=-1)
                with tf.variable_scope("lstm1",reuse=(t!=0)):
                    h0,state1 = self.cell1(x1, state1)

                with tf.variable_scope("attention"):
                    _,x2,_ = self.capsule_attention(h0,flattened_ctx,self.b,self.num_boxes,self.cap_iter)

                with tf.variable_scope("lstm2",reuse=(t!=0)):
                    h1,state2 = self.cell2(x2, state2)
                #concat_h_attention = tf.concat([x2,h1],axis=-1)
                concat_h_attention = h1
                with tf.variable_scope("logits",reuse=(t!=0)):
                    logit = slim.fully_connected(concat_h_attention, self.vocab_size+1, activation_fn=None, scope='logit')

                self.logits.append(logit)


            with tf.variable_scope("loss"):
                loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
                    self.logits,
                    [tf.squeeze(label, [1]) for label in
                     tf.split(axis=1, num_or_size_splits=self.seq_length + 1, value=self.labels[:, 1:])],
                    # self.labels[:,1:] is the target; ignore the first start token
                    [tf.squeeze(mask, [1]) for mask in
                     tf.split(axis=1, num_or_size_splits=self.seq_length + 1, value=self.masks[:, 1:])])
                self.cost = tf.reduce_mean(loss)



        self.lr = tf.Variable(0.0, trainable=False)

        # Collect the rnn variables, and create the optimizer of rnn
        tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='rnnlm')
        grads = utils.clip_by_value(tf.gradients(self.cost, tvars), -self.opt.grad_clip, self.opt.grad_clip)
        optimizer = utils.get_optimizer(self.opt, self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

        tf.summary.scalar('training loss', self.cost)
        tf.summary.scalar('learning rate', self.lr)
        # tf.summary.scalar('cnn learning rate', self.cnn_lr)
        self.summaries = tf.summary.merge_all()

    def build_generator(self):
        """
        Generator for generating captions
        Support sample max or sample from distribution
        No Beam search here; beam search is in decoder
        """
        # Variables for the sample setting
        #self.sample_max = tf.Variable(True, trainable=False, name="sample_max")
        #self.sample_temperature = tf.Variable(1.0, trainable=False, name="temperature")

        self.generator = []
        with tf.variable_scope("rnnlm"):
            flattened_ctx = self.features

            tf.get_variable_scope().reuse_variables()

            initial_state1 = utils.get_initial_state(self.avgFeat, self.cell1.state_size,scope="init_state1")
            initial_state2 = utils.get_initial_state(self.avgFeat, self.cell2.state_size,scope="init_state2")

            # projected context
            # This is used in attention module; do this outside the loop to reduce redundant computations
            # with tf.variable_scope("attention"):

            rnn_input = tf.nn.embedding_lookup(self.Wemb, tf.zeros([self.batch_size], tf.int32))

            outputs = []
            self.alpha_list = []

            h0 = utils.last_hidden_vec(initial_state1)
            h1 = utils.last_hidden_vec(initial_state2)
            state1 = initial_state1
            state2 = initial_state2
            for ind in range(MAX_STEPS):
                if ind > 0:
                    # Reuse the variables after the first timestep.
                    tf.get_variable_scope().reuse_variables()
                x1 = tf.concat([h1,self.avgFeat,rnn_input],axis=-1)
                with tf.variable_scope("lstm1",reuse=(ind!=0)):
                    h0, state1 = self.cell1(x1, state1)
                with tf.variable_scope("attention"):
                    _,x2,c_IJ_list = self.capsule_attention(h0,flattened_ctx,self.b,self.num_boxes,self.cap_iter)
                self.alpha_list.append(c_IJ_list)
                
                with tf.variable_scope("lstm2",reuse=(ind!=0)):
                    h1,state2 = self.cell2(x2, state2)

                #concat_h_attention = tf.concat([x2,h1],axis=-1)
                concat_h_attention = h1
                #print concat_h_attention
                outputs.append(concat_h_attention)
                with tf.variable_scope("logits",reuse=(ind!=0)):
                    prev_logit = slim.fully_connected(concat_h_attention, self.vocab_size+1, activation_fn=None, scope='logit')
                """
                prev_symbol = tf.stop_gradient(tf.cond(self.sample_max,
                                                       lambda: tf.argmax(prev_logit, 1),
                                                       # pick the word with largest probability as the input of next time step
                                                       lambda: tf.squeeze(
                                                           tf.multinomial(
                                                               tf.nn.log_softmax(prev_logit) / self.sample_temperature,
                                                               1), 1)))  # Sample from the distribution"""
                prev_symbol = tf.argmax(prev_logit, 1)
                self.generator.append(prev_symbol)
                rnn_input = tf.nn.embedding_lookup(self.Wemb, prev_symbol)

            #self.g_output = output = tf.reshape(tf.concat(axis=1, values=outputs), [-1, self.rnn_size]) 
            #with tf.variable_scope("logits",reuse=True):
            #    self.g_logits = logits = slim.fully_connected(output, self.vocab_size + 1, activation_fn=None,
            #                                                  scope='logit')
            #self.g_probs = probs = tf.reshape(tf.nn.softmax(logits), [self.batch_size, MAX_STEPS, self.vocab_size + 1])

        self.generator = tf.transpose(tf.reshape(tf.concat(axis=0, values=self.generator), [MAX_STEPS, -1]))

    def decode(self, feature, beam_size, sess, max_steps=MAX_STEPS):
        good_sentences = [] # store sentences already ended with <bos>
        cur_best_cand = [] # store current best candidates
        highest_score = 0.0 # hightest log-likelihodd in good sentences

        cand = {'indexes': [], 'score': 0}
        cur_best_cand.append(cand)

        for i in xrange(max_steps + 1):
            # expand candidates
            cand_pool = []
            #for cand in cur_best_cand:
                #probs, state = self.get_probs_cont(cand['state'], cand['indexes'][-1], sess)
            if i == 0:
                all_probs, all_states1,all_states2 = self.get_probs_init(feature, sess)
            else:
                states1 = [np.vstack([cand['state1'][i] for cand in cur_best_cand]) for i in xrange(len(cur_best_cand[0]['state1']))]
                states2 = [np.vstack([cand['state2'][i] for cand in cur_best_cand]) for i in xrange(len(cur_best_cand[0]['state2']))]
                indexes = [cand['indexes'][-1] for cand in cur_best_cand]
                features = np.vstack([feature] * len(cur_best_cand))
                all_probs, all_states1,all_states2 = self.get_probs_cont(states1,states2, features, indexes, sess)

            for ind_cand in range(len(cur_best_cand)):
                cand = cur_best_cand[ind_cand]
                probs = all_probs[ind_cand]
                state1 = [x[ind_cand] for x in all_states1]
                state2 = [x[ind_cand] for x in all_states2]

                
                probs = np.squeeze(probs)
                probs_order = np.argsort(-probs)
                for ind_b in xrange(beam_size):
                    cand_e = copy.deepcopy(cand)
                    cand_e['indexes'].append(probs_order[ind_b])
                    cand_e['score'] -= np.log(probs[probs_order[ind_b]])
                    cand_e['state1'] = state1
                    cand_e['state2'] = state2
                    cand_pool.append(cand_e)
           
            # get final cand_pool
            cur_best_cand = sorted(cand_pool, key=lambda cand: cand['score'])
            cur_best_cand = utils.truncate_list(cur_best_cand, beam_size)

            # move candidates end with <eos> to good_sentences or remove it
            cand_left = []
            for cand in cur_best_cand:
                if len(good_sentences) > beam_size and cand['score'] > highest_score:
                    continue # No need to expand that candidate
                if cand['indexes'][-1] == 0: #end of sentence
                    good_sentences.append(cand)
                    highest_score = max(highest_score, cand['score'])
                else:
                    cand_left.append(cand)
            cur_best_cand = cand_left
            if not cur_best_cand:
                break
           
    
        # Add candidate left in cur_best_cand to good sentences 
        for cand in cur_best_cand:
            if len(good_sentences) > beam_size and cand['score'] > highest_score:
                continue
            if cand['indexes'][-1] != 0:
                cand['indexes'].append(0)
            good_sentences.append(cand)
            highest_score = max(highest_score, cand['score'])
            
        # Sort good sentences and return the final list
        good_sentences = sorted(good_sentences, key=lambda cand: cand['score'])
        good_sentences = utils.truncate_list(good_sentences, beam_size)
        
        return [sent['indexes'] for sent in good_sentences]

    def build_decoder_rnn(self, first_step):
        with tf.variable_scope("rnnlm"):
            #with tf.variable_scope("feature_proj"):
            #    self.fc = self.fully_connect_layer(self.features, 36,2048,self.att_size)
            flattened_ctx = self.features
            batch_size = tf.shape(self.features)[0]
            tf.get_variable_scope().reuse_variables()

            if not first_step:
                initial_state1 = utils.get_placeholder_state(self.cell1.state_size,scope="placeholder_state1")
                self.decoder_flattened_state1 = utils.flatten_state(initial_state1)  
                initial_state2 = utils.get_placeholder_state(self.cell2.state_size,scope="placeholder_state2")
                self.decoder_flattened_state2 = utils.flatten_state(initial_state2)
            else:
                initial_state1 = utils.get_initial_state(self.avgFeat, self.cell1.state_size,scope="init_state1")
                initial_state2 = utils.get_initial_state(self.avgFeat, self.cell2.state_size,scope="init_state2")
                
            self.decoder_prev_word = tf.placeholder(tf.int32, [None])

            if first_step:
                embed = tf.nn.embedding_lookup(self.Wemb, tf.zeros([batch_size], tf.int32))
            else:
                embed = tf.nn.embedding_lookup(self.Wemb, self.decoder_prev_word)
            h0 = utils.last_hidden_vec(initial_state1)
            h1 = utils.last_hidden_vec(initial_state2)
            

            x1 = tf.concat([h1,self.avgFeat,embed],axis=-1)
            with tf.variable_scope("lstm1"):
                h0,state1 = self.cell1(x1, initial_state1)
            #alpha = self.get_attention(h0)
            #weight_feature =  tf.reduce_sum(self.features*tf.expand_dims(alpha,2),1)
            with tf.variable_scope("attention"):
                _,x2,_ = self.capsule_attention(h0,flattened_ctx,self.b,self.num_boxes,self.cap_iter)

            with tf.variable_scope("lstm2"):
                h1,state2 = self.cell2(x2, initial_state2)
            
            
            concat_h_attention = h1
            with tf.variable_scope("logits"):
                logits = slim.fully_connected(concat_h_attention, self.vocab_size+1, activation_fn=None, scope='logit')

            decoder_probs = tf.reshape(tf.nn.softmax(logits), [batch_size, self.vocab_size+1])

            decoder_state1 = utils.flatten_state(state1)
            decoder_state2 = utils.flatten_state(state2)
            
        return (decoder_probs, decoder_state1, decoder_state2)
    def build_decoder(self):
        self.decoder_model_init = self.build_decoder_rnn(True)
        self.decoder_model_cont = self.build_decoder_rnn(False)

    def get_probs_init(self, features, sess):
        """Use the model to get initial logit"""
        m = self.decoder_model_init
        
        probs, state1,state2 = sess.run(m, {self.images: features,
                                            self.b:np.zeros([1,36,1,1],dtype=float)})
                                                            
        return (probs, state1, state2)
    def get_probs_cont(self, prev_state1,prev_state2, features, prev_word, sess):
        """Use the model to get continued logit"""
        m = self.decoder_model_cont
        prev_word = np.array(prev_word, dtype='int32')
        placeholders = [self.images, self.decoder_prev_word,self.b] + self.decoder_flattened_state1+self.decoder_flattened_state2
        feeded = [features, prev_word,np.zeros([len(prev_word),36,1,1],dtype=float)] + prev_state1+prev_state2

        probs, state1,state2 = sess.run(m, {placeholders[i]: feeded[i] for i in xrange(len(placeholders))})

        return (probs, state1,state2)

        

 







