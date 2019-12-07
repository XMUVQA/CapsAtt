import tensorflow as tf
import config
# functions to create weight and bias
def weight_variable(shape,dtype=tf.float32,name=None,lamda=config.WEIGHT_DECAY):
    var=tf.get_variable(name,shape,dtype=dtype,initializer=tf.contrib.layers.xavier_initializer())
    tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(lamda)(var)) #used for doing weight regularization
    return var

def bias_variable(shape,dtype=tf.float32,name=None):
    return tf.get_variable(name,shape,dtype=dtype,initializer=tf.constant_initializer(0.0))

class CapsAtt(object):


    def __init__(self,mode='train'):

        self.V=config.CNN_DIM #CNN Dimension, default is 2048
        self.A=config.ATTENT_DIM # Attention dimension
        self.H=config.QUESTION_RNN_DIM # RNN Dimension
        self.F=config.FF_DIM
        self.E=config.GLOVE_DIM # Word Embedding dimension
        self.O=config.ANSWER_DIM # Output dimension
        self.Q=config.QUESTION_MAX_LENGTH # Max Question Length.
        self.R=config.CNN_REGION_NUM # image region number
        if mode=='train':
            self.dropout=True
        else:
            self.dropout=False
        self.keep_prob=tf.placeholder(tf.float32,[]) # dropout rate
        self.build_vqa_module()
    def ff_layer(self,x_in,in_dim,out_dim,name_scope,active_function='relu'):
        with tf.variable_scope(name_scope):
            W=weight_variable([in_dim,out_dim],name='ff_weights')
            b=bias_variable([out_dim],name='ff_bias')
            x_out=tf.nn.xw_plus_b(x_in,W,b)
            tf.nn.l2_normalize(x_out,dim=-1)
            if active_function=='relu':
                x_out=tf.nn.relu(x_out)
            elif active_function=='sigmod':
                x_out=tf.nn.sigmoid(x_out)
            elif active_function=='tanh':
                x_out=tf.nn.tanh(x_out)
            elif active_function=='linear':
                x_out=x_out
            return x_out


    def build_vqa_module(self):
        self.img_vec=tf.placeholder(tf.float32,[None,36,2048],'conv_features') # visual features
        self.q_input=tf.placeholder(tf.float32,[None,self.Q,self.E]) # Question Feature, N, step, emb_dim
        self.ans=tf.placeholder(tf.float32,[None,self.O]) # Answer Vector Input.
        self.seqlen=tf.placeholder(tf.int32,[None]) # Sentence lengths
        self.b = tf.placeholder(tf.float32, [None, self.R, 1, 1])  # log priors

        # Adding fake regions, used for not where to look
        fake_region=weight_variable([1,self.V],name='fake_region')
        fake_region=tf.expand_dims(fake_region,axis=0) # 1,1,2048
        fake_region=tf.tile(fake_region,[config.BATCH_SIZE,1,1]) # batch_size, 1, 2048

        Feature_matrix=tf.nn.l2_normalize(self.img_vec,dim=-1)
        #if self.dropout:
        #    F=tf.nn.dropout(Feature_matrix,keep_prob=self.keep_prob)
         #   fake_region=tf.nn.dropout(fake_region,keep_prob=self.keep_prob)
        # F: visual feature matrix
        Feature_matrix = tf.concat([Feature_matrix,fake_region],axis=1) # N, 37, 2048

        # Obtain question feature
        batch_size = tf.shape(self.q_input)[0]
        question_feature = self.build_language_module(self.q_input, self.seqlen, batch_size,mode='lstm')
        with tf.variable_scope('dynamic_capsule_attention'):
            self.out_capsule,self.att_values,self.att_list=self.capsule_attention(Feature_matrix,question_feature,self.b)
        self.out_capsule=self.ff_layer(self.out_capsule,self.A,self.F,'out_capsule_proj')
        if self.dropout:
            self.out_capsule=tf.nn.dropout(self.out_capsule,keep_prob=self.keep_prob)
        self.ff=self.ff_layer(self.out_capsule,self.F,self.F,'foward_layer')
        if self.dropout:
            self.ff=tf.nn.dropout(self.ff,keep_prob=self.keep_prob)
        self.ff2=self.ff_layer(self.ff,self.F,self.F,'foward_layer_two')
        if self.dropout:
            self.ff2=tf.nn.dropout(self.ff2,keep_prob=self.keep_prob)
        self.logits = self.ff_layer(self.ff2,self.F,self.O,active_function='linear',name_scope='prediction_layer')
        self.predict = tf.argmax(tf.nn.softmax(self.logits),axis=1)
        with tf.name_scope('cross_entrophy'):
            self.sigmoid_cross_entrophy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.ans))*self.O # Averaged cross entrophy loss
            self.softmax_cross_entrophy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.ans))
        self.overall_loss = self.sigmoid_cross_entrophy
        correct_prediction = tf.equal(self.predict, tf.argmax(self.ans, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        self.accuracy = accuracy
        

    def capsule_attention(self,F,f_q,b_IJ,iter_num=config.ITER_NUM):

        F=tf.reshape(F,[-1,self.V])
        W_v=weight_variable([self.V,self.A],name='caps_weight')
        W_q=weight_variable([self.H,self.A],name='q_proj')
        f_q=tf.matmul(f_q,W_q)
        F_p=tf.matmul(F,W_v)
        if self.dropout:
            F_p=tf.nn.dropout(F_p,keep_prob=self.keep_prob)

        F_p=tf.reshape(F_p,[-1,self.R,self.A])  ## Reshape F_p [N,49, 1024(VP)
        F_p = tf.expand_dims(F_p, axis=3)  # N,49,1024,1
        F_p_stooped = tf.stop_gradient(F_p, name='F_p_stop_gradient')
        h_out=tf.expand_dims(f_q,1) # [N,1,1024]
        h_out=tf.expand_dims(h_out,3)
        c_ij_list=[]

        with tf.variable_scope('routing'):
            for r_iter in range(iter_num):
                with tf.variable_scope('iter_'+str(r_iter)):
                    c_IJ=tf.nn.softmax(b_IJ,dim=1)  # Based on b_IJ, Obtain c_IJ with shape [N,49,1,1]
                    c_ij_list.append(c_IJ)
                    if r_iter==iter_num-1:
                        f_v=tf.multiply(c_IJ,F_p)  # [N,49,1024,1]
                        f_v=tf.reduce_sum(f_v,axis=1,keep_dims=True)  # shape [batch_size,1,1024,1]
                        h_out=h_out+f_v
                    else:
                        h_out_tile=tf.tile(h_out,[1,self.R,1,1])
                        u_produce_v=tf.matmul(F_p_stooped,h_out_tile,transpose_a=True)
                        b_IJ+=u_produce_v
                        f_v=tf.multiply(c_IJ,F_p_stooped)
                        f_v=tf.reduce_sum(f_v,axis=1,keep_dims=True)
                        h_out=h_out+f_v
        h_out=tf.reshape(h_out,[-1,self.A])
        return h_out,c_IJ,c_ij_list

    def build_language_module(self,word_embs,seqlen,batch_size,mode='bi_gru'):
        if mode=='bi_gru':
            return self.build_bi_gru_modules(word_embs,seqlen,batch_size)
        elif mode=='lstm':
            return self.build_lstm_modules(word_embs,seqlen,batch_size)
        else:
            return self.build_gru_module(word_embs,seqlen,batch_size)
    def build_gru_module(self,word_embs,seqlen,batch_size):
        x = word_embs

        with tf.variable_scope('question_module'):
            cell = tf.nn.rnn_cell.GRUCell(self.H)
            _init_state = cell.zero_state(batch_size, dtype=tf.float32)

            if self.dropout:
                cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.keep_prob)
            outputs, states = tf.nn.dynamic_rnn(
                cell=cell,
                inputs=x,
                dtype=tf.float32,
                sequence_length=seqlen,
                initial_state=_init_state
            )
            return states


    def build_lstm_modules(self,word_embs,seqlen,batch_size):

        """
        Build one layer lstm, and obtain question feature f_q
        :param x_ids:
        :param seqlen:
        :param batch_size:
        :return:
        """
        x=word_embs

        with tf.variable_scope('question_module'):
            lstm_cell=tf.nn.rnn_cell.LSTMCell(self.H)
            _init_state=lstm_cell.zero_state(batch_size,dtype=tf.float32)

            if self.dropout:
                lstm_cell=tf.nn.rnn_cell.DropoutWrapper(lstm_cell,output_keep_prob=self.keep_prob)
            outputs,states=tf.nn.dynamic_rnn(
                cell=lstm_cell,
                inputs=x,
                dtype=tf.float32,
                sequence_length=seqlen,
                initial_state=_init_state
            )
            return states[1]


    def build_bi_gru_modules(self,word_embs,seqlen,batch_size):

        x=word_embs

        with tf.variable_scope('question_module'):
            fw_cell=tf.nn.rnn_cell.GRUCell(self.H/2)
            bw_cell=tf.nn.rnn_cell.GRUCell(self.H/2)
            _init_fw_state = fw_cell.zero_state(batch_size, dtype=tf.float32)
            _init_bw_state = bw_cell.zero_state(batch_size, dtype=tf.float32)

            if self.dropout:
                fw_cell=tf.nn.rnn_cell.DropoutWrapper(fw_cell,output_keep_prob=self.keep_prob)
                bw_cell=tf.nn.rnn_cell.DropoutWrapper(bw_cell,output_keep_prob=self.keep_prob)

            outputs,states=tf.nn.bidirectional_dynamic_rnn(
                cell_fw=fw_cell,
                cell_bw=bw_cell,
                inputs=x,
                dtype=tf.float32,
                initial_state_fw= _init_fw_state,
                initial_state_bw= _init_bw_state,
                sequence_length=seqlen
            )

            f_q=tf.concat(states,axis=1)

            return f_q
