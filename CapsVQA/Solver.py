__author__ = 'antony'
import config
import tensorflow as tf
import time,json,os
import sys
from visulize_attention import *
from data_loader import VQADataLoader
sys.path.append(config.VQA_TOOLS_PATH)
sys.path.append(config.VQA_EVAL_TOOLS_PATH)
from vqaTools.vqa import VQA
from vqaEvaluation.vqaEval import VQAEval


class Solver(object):

    def __init__(self,model):

        self.model=model
        self.global_epoch_step=0
        self.lr=tf.placeholder(tf.float32)
        self.vqa_optimizer=tf.train.AdamOptimizer(self.lr)

    def save_vqa_model(self,sess,scope,step,update=False):
        path=config.FOLDER_NAME+'/models/'
        saver=tf.train.Saver(tf.get_collection(tf.GraphKeys.VARIABLES,scope=scope))
        print 'saving model to %s'%path
        if update:
            saver.save(sess,path+config.MODEL_NAME)
        else:
            saver.save(sess,path+config.MODEL_NAME,step)


    def idx2answers(self,predicts):
        return [self.dataloader.idx2ans[str(x)] for x in predicts]

    def load_vqa_model(self,sess,scope):
        path = config.FOLDER_NAME + '/models/'
        saver=tf.train.Saver(tf.get_collection(tf.GraphKeys.VARIABLES,scope=scope))
        saver.restore(sess,path+config.MODEL_LOAD_NAME)

    def train(self):
        #################### Basic Operations#########################
        LR=config.LEARNING_RATE
        overall_loss=self.model.overall_loss
        cross_entrophy=self.model.sigmoid_cross_entrophy
        accuracy=self.model.accuracy
        vqa_predict=self.model.predict
        log_path = config.FOLDER_NAME + '/logs/'
        vqa_train_op=self.vqa_optimizer.minimize(overall_loss)
        result_file_path = log_path + 'val_results.txt'
        if os.path.exists(result_file_path):
            result_file = open(result_file_path, 'a')
        else:
            result_file = open(result_file_path, 'w')
        #################### Summary Setting #########################
        val_average_loss=tf.placeholder('float')
        val_average_accuracy=tf.placeholder('float')

        train_ol_summary=tf.summary.scalar('train_overall_loss',overall_loss)
        train_ce_summary=tf.summary.scalar('train_cross_entrophy',cross_entrophy)
        train_acc_summary=tf.summary.scalar('accuracy',accuracy)

        val_average_l_summary=tf.summary.scalar('val_ave_cross_entrophy',val_average_loss)
        val_average_acc_summary=tf.summary.scalar('val_average_accuracy',val_average_accuracy)



        #################### Session Setting#########################
        sess=tf.Session(config=tf.ConfigProto(log_device_placement=False))
        sess.run(tf.global_variables_initializer())
        train_writer=tf.summary.FileWriter(log_path+'train',sess.graph)
        valid_writer=tf.summary.FileWriter(log_path+'valid')
        if config.USE_PRE_VQA_MODEL:
            self.load_vqa_model(sess,config.MODEL_SCOPE)
        #################### Train Step Function#########################

        dataloader=VQADataLoader(batchsize=config.BATCH_SIZE,mode='train')
        start_t = time.time()
        best_result = [0.0, 0]
        val_counter = 0
        results = []
        for i in range(config.MAX_TRAIN_STEP):
            q_strs, q_word_vec_list, q_len_list, ans_vectors, img_features, t_qid_list, img_ids, epoch_counter = dataloader.next_batch(
                config.BATCH_SIZE)
            batch_size=len(t_qid_list)
            if (i+1)%config.DECAY_STEP==0:
                LR=LR*config.DECAY_RATE
                print 'decay learning',LR
            feed_dict = {self.model.q_input: q_word_vec_list, self.model.ans: ans_vectors,
                         self.model.seqlen: q_len_list, self.model.img_vec: img_features,
                         self.lr: LR, self.model.keep_prob: config.KEEP_PROB,
                         self.model.b:np.zeros([batch_size,config.CNN_REGION_NUM,1,1],dtype=float)}
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            _ = sess.run(vqa_train_op, feed_dict=feed_dict)
            to_summary, tc_summary, ta_summary, acc, o_loss, c_loss, predict = sess.run(
                [train_ol_summary, train_ce_summary, train_acc_summary, accuracy, overall_loss, cross_entrophy,
                 vqa_predict], options=run_options,
                run_metadata=run_metadata, feed_dict=feed_dict)
            train_writer.add_summary(to_summary, i)
            train_writer.add_summary(tc_summary, i)
            train_writer.add_summary(ta_summary, i)
            if (i+1)%config.SHOW_STEP==0:
                cost_t=time.time()-start_t
                print '\n Train Process: Step: %d|%d, Overall Loss: %.7f, Sigmoid Cross Entrophy: %.7f, lr: %.7f, accuracy: %.3f, time:%d s' \
                      % (
                      i + 1, config.MAX_TRAIN_STEP, o_loss, c_loss, LR, acc, cost_t)
                print 'Question: %s' % (q_strs[0])
                answer = ans_vectors[0].argmax(axis=0)
                print 'Answer:', dataloader.vec_to_answer(answer),answer
                v_pred = predict[0]
                print 'predict:',dataloader.vec_to_answer(v_pred) , v_pred
                start_t=time.time()
            if (i+1)%config.VAL_STEP==0:
                val_ave_l,val_ave_acc,acc_per_question_type,acc_per_answer_type=self.exec_validation(sess,mode='val',folder=config.FOLDER_NAME)
                vl_summary, va_summary = sess.run([val_average_l_summary, val_average_acc_summary],
                                                  feed_dict={val_average_loss: val_ave_l,
                                                             val_average_accuracy: val_ave_acc})
                print 'the average validation losses is %.7f, average accuracy is %.5f' % \
                      (val_ave_l, val_ave_acc)
                print 'per answer:',acc_per_answer_type
                print 'per quetion type:',acc_per_question_type
                result_file.write('%d\n' % i + str(val_ave_acc) + '\n' + str(acc_per_answer_type) + '\n' + str(
                    acc_per_question_type) + '\n')
                valid_writer.add_summary(vl_summary, i)
                valid_writer.add_summary(va_summary, i)
                if val_ave_acc>best_result[0]:
                    print 'the previous max val acc is', best_result[0], 'at iter',best_result[1]
                    best_result[0]=val_ave_acc
                    best_result[1]=i
                    print 'now the best result is',best_result[0]
                    val_counter=0
                else:
                    print 'the best result is',best_result[0],'at iter',best_result[1]
                    val_counter+=1
                results.append([i,c_loss, val_ave_l, val_ave_acc, acc_per_question_type, acc_per_answer_type])
                best_result_idx = np.array([x[3] for x in results]).argmax()
                print 'Best accuracy of', results[best_result_idx][3], 'was at iteration', results[best_result_idx][0]
            if val_counter>5:
                self.save_vqa_model(sess,config.MODEL_SCOPE,i+1)
                break
            if (i+1)%config.SAVE_STEP==0:
                print 'saving model .......'
                self.save_vqa_model(sess,config.MODEL_SCOPE,i+1) # Save VQA Model

    def exec_validation(self,sess,mode,folder, it=0, visualize=False):

        dp = VQADataLoader(mode=mode, batchsize=config.VAL_BATCH_SIZE, folder=folder)
        total_questions = len(dp.getQuesIds())
        epoch = 0
        pred_list = []
        testloss_list = []
        stat_list = []
        self.model.dropout=False

        while epoch == 0:
            q_strs, q_word_vec_list, q_len_list, ans_vectors, img_features, t_qid_list, img_ids, epoch = dp.next_batch(
                config.BATCH_SIZE)
            batch_size=len(t_qid_list)
            feed_dict = {self.model.q_input: q_word_vec_list, self.model.ans: ans_vectors,
                         self.model.seqlen: q_len_list, self.model.img_vec: img_features,
                         self.model.keep_prob: 1.0,
                         self.model.b:np.zeros([batch_size,config.CNN_REGION_NUM,1,1],dtype=float)}

            t_predict_list,predict_loss=sess.run([self.model.predict, self.model.softmax_cross_entrophy], feed_dict=feed_dict)
            t_pred_str = [dp.vec_to_answer(pred_symbol) for pred_symbol in t_predict_list]
            testloss_list.append(predict_loss)
            ans_vectors=np.asarray(ans_vectors).argmax(1)
            for qid, iid, ans, pred in zip(t_qid_list, img_ids, ans_vectors, t_pred_str):
                pred_list.append((pred, int(dp.getStrippedQuesId(qid))))
                if visualize:
                    q_list = dp.seq_to_list(dp.getQuesStr(qid))
                    if mode == 'test-dev' or 'test':
                        ans_str = ''
                        ans_list = [''] * 10
                    else:
                        ans_str = dp.vec_to_answer(ans)
                        ans_list = [dp.getAnsObj(qid)[i]['answer'] for i in xrange(10)]
                    stat_list.append({'qid': qid,
                        'q_list': q_list,
                        'iid': iid,
                        'answer': ans_str,
                        'ans_list': ans_list,
                        'pred': pred})
            percent = 100 * float(len(pred_list)) / total_questions
            sys.stdout.write('\r' + ('%.2f' % percent) + '%')
            sys.stdout.flush()
        self.model.dropout = True
        print 'Deduping arr of len', len(pred_list)
        deduped = []
        seen = set()
        for ans, qid in pred_list:
            if qid not in seen:
                seen.add(qid)
                deduped.append((ans, qid))
        print 'New len', len(deduped)
        final_list = []
        for ans, qid in deduped:
            final_list.append({u'answer': ans, u'question_id': qid})

        mean_testloss = np.array(testloss_list).mean()

        if mode == 'val':
            valFile = './%s/val2015_resfile' % folder
            with open(valFile, 'w') as f:
                json.dump(final_list, f)

            annFile = config.DATA_PATHS['val']['ans_file']
            quesFile = config.DATA_PATHS['val']['ques_file']
            vqa = VQA(annFile, quesFile)
            vqaRes = vqa.loadRes(valFile, quesFile)
            vqaEval = VQAEval(vqa, vqaRes, n=2)
            vqaEval.evaluate()
            acc_overall = vqaEval.accuracy['overall']
            acc_perQuestionType = vqaEval.accuracy['perQuestionType']
            acc_perAnswerType = vqaEval.accuracy['perAnswerType']
            return mean_testloss, acc_overall, acc_perQuestionType, acc_perAnswerType
        elif mode == 'test-dev':
            filename = './%s/vqa_OpenEnded_mscoco_test-dev2015_%s-%d' % (folder, folder,it) + str(it).zfill(8) + '_results'
            with open(filename + '.json', 'w') as f:
                json.dump(final_list, f)

        elif mode == 'test':
            filename = './%s/vqa_OpenEnded_mscoco_test2015_%s-%d' % (folder, folder,it) + str(it).zfill(8) + '_results'
            with open(filename + '.json', 'w') as f:
                json.dump(final_list, f)


    def eval(self):
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
        sess.run(tf.global_variables_initializer())
        self.load_vqa_model(sess, config.MODEL_SCOPE)
        self.exec_validation(sess, mode='test-dev', folder=config.FOLDER_NAME)

    def train_eval(self):
        """
        Used to train and eval model. e.g. Using train+val or train+val+genome as training data
        and eval test or test-dev.
        """
        #################### Basic Operations#########################
        overall_loss=self.model.overall_loss
        cross_entrophy=self.model.sigmoid_cross_entrophy
        accuracy=self.model.accuracy
        vqa_predict=self.model.predict
        log_path = config.FOLDER_NAME + '/logs/'
        LR=config.LEARNING_RATE
        vqa_train_op=self.vqa_optimizer.minimize(overall_loss)
        #################### Summary Setting #########################
        train_ol_summary=tf.summary.scalar('train_overall_loss',overall_loss)
        train_ce_summary=tf.summary.scalar('train_cross_entrophy',cross_entrophy)
        train_acc_summary=tf.summary.scalar('accuracy',accuracy)
        #################### Session Setting#########################
        sess=tf.Session(config=tf.ConfigProto(log_device_placement=False))
        sess.run(tf.global_variables_initializer())
        train_writer=tf.summary.FileWriter(log_path+'train',sess.graph)
        if config.USE_PRE_VQA_MODEL:
            self.load_vqa_model(sess,config.MODEL_SCOPE)
        #################### Train Step Function#########################
        dataloader=VQADataLoader(batchsize=config.BATCH_SIZE,mode='train')
        start_t = time.time()
        for i in range(config.MAX_TRAIN_STEP):
            q_strs, q_word_vec_list, q_len_list, ans_vectors, img_features, t_qid_list, img_ids, epoch_counter = dataloader.next_batch(
                config.BATCH_SIZE)
            batch_size=len(t_qid_list)
            feed_dict = {self.model.q_input: q_word_vec_list, self.model.ans: ans_vectors,
                         self.model.seqlen: q_len_list, self.model.img_vec: img_features,
                         self.lr: LR, self.model.keep_prob: config.KEEP_PROB,
                         self.model.b:np.zeros([batch_size,config.CNN_REGION_NUM,1,1],dtype=float)}
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            _ = sess.run(vqa_train_op, feed_dict=feed_dict)
            to_summary, tc_summary, ta_summary, acc, o_loss, c_loss, predict = sess.run(
                [train_ol_summary, train_ce_summary, train_acc_summary, accuracy, overall_loss, cross_entrophy,
                 vqa_predict], options=run_options,
                run_metadata=run_metadata, feed_dict=feed_dict)
            train_writer.add_summary(to_summary, i)
            train_writer.add_summary(tc_summary, i)
            train_writer.add_summary(ta_summary, i)
            if (i+1)%config.DECAY_STEP==0:
                LR=LR*config.DECAY_RATE
            if (i+1)%config.SHOW_STEP==0:
                cost_t=time.time()-start_t
                print '\n Training Process: Step: %d|%d, Overall Loss: %.7f, Sigmoid Cross Entrophy: %.7f, lr: %.7f, accuracy: %.3f, time:%d s' \
                      % (
                      i + 1, config.MAX_TRAIN_STEP, o_loss, c_loss, LR, acc, cost_t)
                print 'Question: %s' % (q_strs[0])
                answer = ans_vectors[0].argmax(axis=0)
                print 'Answer:', dataloader.vec_to_answer(answer),answer
                v_pred = predict[0]
                print 'predict:',dataloader.vec_to_answer(v_pred) , v_pred
                start_t=time.time()
            if  (i+1)%config.VAL_STEP==0:
                self.exec_validation(sess,mode=config.TEST_MODE,it=i,folder=config.FOLDER_NAME)
            if (i+1)%config.SAVE_STEP==0:
                print 'saving model .......'
                self.save_vqa_model(sess,config.MODEL_SCOPE,i+1) # Save VQA Model









