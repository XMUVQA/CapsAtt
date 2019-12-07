from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
import os

import copy

import numpy as np
import show_utils as utils

# The maximimum step during generation
MAX_STEPS = 30

def weight_variable(shape,dtype=tf.float32,name=None,lamda=0.0001):
    var=tf.get_variable(name,shape,dtype=dtype,initializer=tf.contrib.layers.xavier_initializer())
    tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(lamda)(var))
    return var
class FRCNN_CapsCaption_V3_L2():

    def initialize(self, sess):
        # Initialize the variables
        sess.run(tf.global_variables_initializer())
        # Initialize the saver
        self.saver = tf.train.Saver(tf.trainable_variables())
        if self.opt.load_pre_model:
            self.saver.restore(sess, self.opt.pre_model_path)
        if self.opt.eval_or_train =="train":
            self.summary_writer = tf.summary.FileWriter(self.opt.checkpoint_path, sess.graph)
        print ("total paramter num:%d"%utils.get_num_params())

    def __init__(self, opt):
        self.vocab_size = opt.vocab_size
        self.input_encoding_size = opt.input_encoding_size
        self.rnn_size = opt.rnn_size
        self.num_layers = 1
        self.drop_prob_lm = opt.drop_prob_lm
        self.seq_length = opt.seq_length

        self.seq_per_img = opt.seq_per_img
        self.att_hid_size = opt.att_size

        self.opt = opt
        self.region_num=opt.num_boxes
        # Variable indicating in training mode or evaluation mode
        self.training = tf.Variable(True, trainable=False, name="training")

        # Input variables
        self.images = tf.placeholder(tf.float32, [None, self.region_num,2048], name="images")
        self.b_IJ = self.b_IJ = tf.placeholder(tf.float32, [None, self.region_num, 1, 1], name='b_IJ')
        self.labels = tf.placeholder(tf.int32, [None, self.seq_length + 2], name="labels")
        self.masks = tf.placeholder(tf.float32, [None, self.seq_length + 2], name="masks")
        # Build CNN
        self.context=tf.nn.l2_normalize(self.images,axis=-1)

        self.fc7 = tf.reduce_mean(self.context,axis=1) # N,2048

        self.cnn_dim=2048
        # Variable in language model
        with tf.variable_scope("rnnlm"):
            # Word Embedding table
            self.Wemb = tf.Variable(tf.random_uniform([self.vocab_size + 1, self.input_encoding_size], -0.1, 0.1),
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

            # keep_prob is a function of training flag
            self.keep_prob = tf.cond(self.training,
                                     lambda: tf.constant(1 - self.drop_prob_lm),
                                     lambda: tf.constant(1.0), name='keep_prob')

            # basic cell has dropout wrapper
            self.basic_cell = cell = tf.contrib.rnn.DropoutWrapper(cell_fn(self.rnn_size), 1.0, self.keep_prob)
            # cell is the final cell of each timestep
            self.cell = tf.contrib.rnn.MultiRNNCell([cell] * opt.num_layers)

    def get_alpha(self, prev_h, pctx):
        # projected state
        if self.att_hid_size == 0:
            pstate = slim.fully_connected(prev_h, 1, activation_fn=None, scope='h_att')  # (batch * seq_per_img) * 1
            alpha = pctx + tf.expand_dims(pstate, 1)  # (batch * seq_per_img) * 196 * 1
            alpha = tf.squeeze(alpha, [2])  # (batch * seq_per_img) * 196
            alpha = tf.nn.softmax(alpha)
        else:
            pstate = slim.fully_connected(prev_h, self.att_hid_size, activation_fn=None,
                                          scope='h_att')  # (batch * seq_per_img) * att_hid_size
            pctx_ = pctx + tf.expand_dims(pstate, 1)  # (batch * seq_per_img) * 196 * att_hid_size
            pctx_ = tf.nn.tanh(pctx_)  # (batch * seq_per_img) * 196 * att_hid_size
            alpha = slim.fully_connected(pctx_, 1, activation_fn=None, scope='alpha')  # (batch * seq_per_img) * 196 * 1
            alpha = tf.squeeze(alpha, [2])  # (batch * seq_per_img) * 196
            alpha = tf.nn.softmax(alpha)
        return alpha

    def get_caps_alpha(self, prev_h, flattened_ctx, b_IJ, r_num, iter=3):

        F_reshape = tf.reshape(flattened_ctx, [-1, self.cnn_dim])  # N*R, 512
        W_F = weight_variable([self.cnn_dim, self.att_hid_size], name='imag_proj')
        F_p = tf.matmul(F_reshape, W_F)
        F_p = tf.reshape(F_p, [-1, r_num, self.att_hid_size])
        F_p = tf.expand_dims(F_p, axis=3)  # N, R, A, 1
        F_p_stop = tf.stop_gradient(F_p, name='F_p_stop_gradient')

        W_h = weight_variable([self.rnn_size, self.att_hid_size], name='h_proj')
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

        h_out = tf.reshape(h_proj, [-1, self.att_hid_size])
        alpha = tf.reshape(c_IJ, [-1, self.region_num])

        return alpha, h_out, c_IJ_list

    def build_model(self):
        with tf.name_scope("batch_size"):
            # Get batch_size from the first dimension of self.images
            self.batch_size = tf.shape(self.images)[0]
        with tf.variable_scope("rnnlm"):
            # Flatten the context
            flattened_ctx = self.context

            # Initialize the first hidden state with the mean context
            initial_state = utils.get_initial_state(self.fc7, self.cell.state_size)
            # Replicate self.seq_per_img times for each state and image embedding
            self.initial_state = initial_state = utils.expand_feat(initial_state, self.seq_per_img)
            self.flattened_ctx = flattened_ctx = tf.reshape(
                tf.tile(tf.expand_dims(flattened_ctx, 1), [1, self.seq_per_img, 1, 1]),
                [self.batch_size * self.seq_per_img, self.region_num, self.cnn_dim])

            # projected context
            # This is used in attention module; do this outside the loop to reduce redundant computations
            # with tf.variable_scope("attention"):


            rnn_inputs = tf.split(axis=1, num_or_size_splits=self.seq_length + 1,
                                  value=tf.nn.embedding_lookup(self.Wemb, self.labels[:, :self.seq_length + 1]))
            rnn_inputs = [tf.squeeze(input_, [1]) for input_ in rnn_inputs]

            prev_h = utils.last_hidden_vec(initial_state)

            self.alphas = []
            self.alpha_list=[]
            self.logits = []
            outputs = []
            state = initial_state
            for ind in range(self.seq_length + 1):
                if ind > 0:
                    # Reuse the variables after the first timestep.
                    tf.get_variable_scope().reuse_variables()
                
                with tf.variable_scope("attention"):
                    alpha, prev_j ,c_ij_list= self.get_caps_alpha(prev_h, flattened_ctx,self.b_IJ,self.region_num)
                    alpha = tf.reshape(alpha, [-1, self.region_num])
                    self.alphas.append(alpha)
                    #weighted_context = tf.reduce_sum(flattened_ctx * tf.expand_dims(alpha, 2), 1)

                output, state = self.cell(tf.concat(axis=1, values=[prev_j,rnn_inputs[ind]]), state)
                # Save the current output for next time step attention
                prev_h = output
                # Get the score of each word in vocabulary, 0 is end token.
                self.logits.append(slim.fully_connected(output, self.vocab_size + 1, activation_fn=None, scope='logit'))

        with tf.variable_scope("loss"):
            loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
                self.logits,
                [tf.squeeze(label, [1]) for label in
                 tf.split(axis=1, num_or_size_splits=self.seq_length + 1, value=self.labels[:, 1:])],
                # self.labels[:,1:] is the target; ignore the first start token
                [tf.squeeze(mask, [1]) for mask in
                 tf.split(axis=1, num_or_size_splits=self.seq_length + 1, value=self.masks[:, 1:])])
            self.cost = tf.reduce_mean(loss)


        self.final_state = state
        self.lr = tf.Variable(0.0, trainable=False)
        self.cnn_lr = tf.Variable(0.0, trainable=False)

        # Collect the rnn variables, and create the optimizer of rnn
        tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='rnnlm')
        grads = utils.clip_by_value(tf.gradients(self.cost, tvars), -self.opt.grad_clip, self.opt.grad_clip)
        optimizer = utils.get_optimizer(self.opt, self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

        # Collect the cnn variables, and create the optimizer of cnn
        # cnn_tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='cnn')
        # cnn_grads = utils.clip_by_value(tf.gradients(self.cost, cnn_tvars), -self.opt.grad_clip, self.opt.grad_clip)
        # cnn_optimizer = utils.get_cnn_optimizer(self.opt, self.cnn_lr)
        # self.cnn_train_op = cnn_optimizer.apply_gradients(zip(cnn_grads, cnn_tvars))

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
        self.sample_max = tf.Variable(True, trainable=False, name="sample_max")
        self.sample_temperature = tf.Variable(1.0, trainable=False, name="temperature")

        self.generator = []
        with tf.variable_scope("rnnlm"):
            flattened_ctx = self.context

            tf.get_variable_scope().reuse_variables()

            initial_state = utils.get_initial_state(self.fc7, self.cell.state_size)

            # projected context
            # This is used in attention module; do this outside the loop to reduce redundant computations
            # with tf.variable_scope("attention"):

            rnn_input = tf.nn.embedding_lookup(self.Wemb, tf.zeros([self.batch_size], tf.int32))

            prev_h = utils.last_hidden_vec(initial_state)

            self.g_alphas = []
            self.g_alpha_list=[]
            outputs = []
            state = initial_state
            for ind in range(MAX_STEPS):

                
                with tf.variable_scope("attention"):
                    alpha, prev_j,c_ij_list = self.get_caps_alpha(prev_h, flattened_ctx,self.b_IJ,self.region_num)
                    self.g_alpha_list.append(c_ij_list)
                    self.g_alphas.append(alpha)
                    #weighted_context = tf.reduce_sum(flattened_ctx * tf.expand_dims(alpha, 2), 1)

                output, state = self.cell(tf.concat(axis=1, values=[prev_j,rnn_input]), state)
                prev_h = output
                outputs.append(output)
                # Get the input of next timestep
                prev_logit = slim.fully_connected(output, self.vocab_size + 1, activation_fn=None, scope='logit')
                prev_symbol = tf.stop_gradient(tf.cond(self.sample_max,
                                                       lambda: tf.argmax(prev_logit, 1),
                                                       # pick the word with largest probability as the input of next time step
                                                       lambda: tf.squeeze(
                                                           tf.multinomial(
                                                               tf.nn.log_softmax(prev_logit) / self.sample_temperature,
                                                               1), 1)))  # Sample from the distribution
                self.generator.append(prev_symbol)
                rnn_input = tf.nn.embedding_lookup(self.Wemb, prev_symbol)

            self.g_output = output = tf.reshape(tf.concat(axis=1, values=outputs), [-1,
                                                                                    self.rnn_size])  # outputs[1:], because we don't calculate loss on time 0.
            self.g_logits = logits = slim.fully_connected(output, self.vocab_size + 1, activation_fn=None,
                                                          scope='logit')
            self.g_probs = probs = tf.reshape(tf.nn.softmax(logits), [self.batch_size, MAX_STEPS, self.vocab_size + 1])

        self.generator = tf.transpose(tf.reshape(tf.concat(axis=0, values=self.generator), [MAX_STEPS, -1]))

   
    def build_decoder_rnn(self, first_step):
        with tf.variable_scope("rnnlm"):
            flattened_ctx = self.context

            tf.get_variable_scope().reuse_variables()

            if not first_step:
                initial_state = utils.get_placeholder_state(self.cell.state_size)
                self.decoder_flattened_state = utils.flatten_state(initial_state)
            else:
                initial_state = utils.get_initial_state(self.fc7, self.cell.state_size)

            self.decoder_prev_word = tf.placeholder(tf.int32, [None])

            if first_step:
                rnn_input = tf.nn.embedding_lookup(self.Wemb, tf.zeros([self.batch_size], tf.int32))
            else:
                rnn_input = tf.nn.embedding_lookup(self.Wemb, self.decoder_prev_word)

            # projected context
            # This is used in attention module; do this outside the loop to reduce redundant computations
            # with tf.variable_scope("attention"):


            alphas = []
            alpha_list=[]
            outputs = []
            prev_h = utils.last_hidden_vec(initial_state)
            with tf.variable_scope("attention"):
                alpha, prev_j,c_ij_list = self.get_caps_alpha(prev_h, flattened_ctx,self.b_IJ,self.region_num)
                alpha_list.append(c_ij_list)
                alphas.append(alpha)

            output, state = self.cell(tf.concat(axis=1, values=[prev_j,rnn_input]), initial_state)
            outputs.append(output)
            logits = slim.fully_connected(output, self.vocab_size + 1, activation_fn=None, scope='logit')
            
            decoder_probs = tf.reshape(tf.nn.softmax(logits), [self.batch_size, self.vocab_size + 1])
            decoder_state = utils.flatten_state(state)
        return [decoder_probs, decoder_state]

    def build_decoder(self):
        self.decoder_model_init = self.build_decoder_rnn(True)
        self.decoder_model_cont = self.build_decoder_rnn(False)

    def decode(self, img, beam_size, sess, b_IJ, max_steps=MAX_STEPS):
        """Decode an image with a sentences."""

        # Initilize beam search variables
        # Candidate will be represented with a dictionary
        #   "indexes": a list with indexes denoted a sentence;
        #   "words": word in the decoded sentence without <bos>
        #   "score": log-likelihood of the sentence
        #   "state": RNN state when generating the last word of the candidate
        good_sentences = []  # store sentences already ended with <bos>
        cur_best_cand = []  # store current best candidates
        highest_score = 0.0  # hightest log-likelihodd in good sentences

        # Get the initial logit and state
        cand = {'indexes': [], 'score': 0}
        cur_best_cand.append(cand)
        # print ('number of layers',self.num_layers)
        # Expand the current best candidates until max_steps or no candidate
        for i in xrange(max_steps + 1):
            # expand candidates
            cand_pool = []
            # for cand in cur_best_cand:
            # probs, state = self.get_probs_cont(cand['state'], cand['indexes'][-1], sess)
            if i == 0:
                all_probs, all_states = self.get_probs_init(img, sess, b_IJ)
                # print ('len of first probs', len(all_probs))
            else:
                states = [np.vstack([cand['state'][i] for cand in cur_best_cand]) for i in
                          xrange(len(cur_best_cand[0]['state']))]
                # print ('----state shape', len(states),states[0].shape)
                indexes = [cand['indexes'][-1] for cand in cur_best_cand]
                imgs = np.vstack([img] * len(cur_best_cand))
                # print ('----img shape,',imgs.shape)
                b_IJs = np.vstack([b_IJ]*len(cur_best_cand))
                all_probs, all_states = self.get_probs_cont(states, imgs, indexes, sess, b_IJs)
            for ind_cand in range(len(cur_best_cand)):
                cand = cur_best_cand[ind_cand]
                probs = all_probs[ind_cand]
                state = [x[ind_cand] for x in all_states]

                probs = np.squeeze(probs)
                probs_order = np.argsort(-probs)
                for ind_b in xrange(beam_size):
                    cand_e = copy.deepcopy(cand)
                    cand_e['indexes'].append(probs_order[ind_b])
                    cand_e['score'] -= np.log(probs[probs_order[ind_b]])
                    cand_e['state'] = state
                    cand_pool.append(cand_e)
            # get final cand_pool
            cur_best_cand = sorted(cand_pool, key=lambda cand: cand['score'])
            cur_best_cand = utils.truncate_list(cur_best_cand, beam_size)

            # move candidates end with <eos> to good_sentences or remove it
            cand_left = []
            for cand in cur_best_cand:
                if len(good_sentences) > beam_size and cand['score'] > highest_score:
                    continue  # No need to expand that candidate
                if cand['indexes'][-1] == 0:  # end of sentence
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

    def get_probs_init(self, img, sess, b_IJ):
        """Use the model to get initial logit"""
        m = self.decoder_model_init

        probs, state = sess.run(m, {self.images: img, self.b_IJ: b_IJ})
        # print ('state shape from m',state)
        return (probs, state)

    def get_probs_cont(self, prev_state, img, prev_word, sess, b_IJ):
        """Use the model to get continued logit"""
        m = self.decoder_model_cont
        prev_word = np.array(prev_word, dtype='int32')

        # Feed images, input words, and the flattened state of previous time step.
        placeholders = [self.images, self.b_IJ, self.decoder_prev_word] + self.decoder_flattened_state
        feeded = [img, b_IJ, prev_word] + prev_state

        probs, state = sess.run(m, {placeholders[i]: feeded[i] for i in xrange(len(placeholders))})

        return (probs, state)
