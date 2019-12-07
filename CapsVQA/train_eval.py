__author__ = 'antony'
from  Solver import Solver
from Model import CapsAtt
import tensorflow as tf
import config,os
import json
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']=config.GPU_ID
from utils import make_vocab_files

def main():
    folder = config.FOLDER_NAME
    if not os.path.exists('./%s' % folder):
        os.makedirs('./%s' % folder)
    if not os.path.exists('./%s/models' % folder):
        os.makedirs('./%s/models/' % folder)
    if not os.path.exists('./%s/logs' % folder):
        os.makedirs('./%s/logs/' % folder)
    if os.path.exists('./%s/vdict.json' % folder) and os.path.exists('./%s/adict.json' % folder):
        print 'restoring vocab'
        with open('./%s/vdict.json' % folder, 'r') as f:
            question_vocab = json.load(f)
        with open('./%s/adict.json' % folder, 'r') as f:
            answer_vocab = json.load(f)
    else:
        question_vocab, answer_vocab = make_vocab_files()
        with open('./%s/vdict.json' % folder, 'w') as f:
            json.dump(question_vocab, f)
        with open('./%s/adict.json' % folder, 'w') as f:
            json.dump(answer_vocab, f)
    print 'question vocab size:', len(question_vocab)
    print 'answer vocab size:', len(answer_vocab)
    with tf.variable_scope(config.MODEL_SCOPE):
        model = CapsAtt()
    solver = Solver(model)
    solver.train_eval()
if __name__ == '__main__':
	main()
