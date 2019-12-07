 # -*- coding: utf-8 -*
import argparse


#embedding input：10010  output：1000
#hidden_pre 1000
#lstm_cell 1000
#vocab_size :10010


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_or_train', type=str, default="train",
                    help='eval,train,test,viz')#验证，训练，测试，可视化
    parser.add_argument('--load_pre_model', type=bool, default=False,
                    help='load pre model or not')#是否导入已训练模型

    parser.add_argument('--cnn_weight', type=str, default='/home/cwq/extra_disk1/study/neuraltalk2-tensorflow/models/vgg16.npy',
                    help='path to CNN tf model. Note this MUST be a vgg16 right now.')#CNN网络的权重，如vgg16的权重

    #model settings
    parser.add_argument('--pre_model_path', type=str, default="",
                    help='pre model path')#已训练模型的路径
    parser.add_argument('--cnn_model', type=str, default='frcnn',
                    help='frcnn vgg16 vgg19 resnet')#采用的视觉特征网络，frcnn ,vgg16,resnet

    parser.add_argument('--num_boxes', type=int, default=36,
                    help='the rpn feature num 36 or 100 in frcnn,or the num boxes in cnn model,may be is 49/196')#对应的区域数，frcnn(36/100)  vgg/resnet  49/196  这里为196.
    #model settings
    parser.add_argument('--rnn_size', type=int, default=1000,
                    help='size of the rnn in number of hidden nodes in each layer')

    parser.add_argument('--batch_size', type=int, default=10,
                    help='minibatch size')

    parser.add_argument('--cap_iter', type=int, default=3,
                    help='capsule attentin iter num')


    parser.add_argument('--input_encoding_size', type=int, default=1000,
                    help='the encoding size of each token in the vocabulary, and the image.')
    
    parser.add_argument('--num_layers', type=int, default=1,
                    help='number of layers in the RNN')

    parser.add_argument('--rnn_type', type=str, default='lstm',
                    help='rnn, gru, or lstm')

    parser.add_argument('--seq_length', type=int, default=20,
                    help='max length of seq in the caption')

    parser.add_argument('--att_size', type=int, default=512,
                    help='the dim in attention')


    parser.add_argument('--seq_per_img', type=int, default=5,
                    help='number of captions to sample for each image during training. Done for efficiency since CNN forward pass is expensive. E.g. coco has 5 sents/image')


    parser.add_argument('--drop_prob_lm', type=float, default=0.5,
                    help='strength of dropout in the Language Model RNN')

    parser.add_argument('--grad_clip', type=float, default=0.1, #5.,
                    help='clip gradients at this value')

    parser.add_argument('--checkpoint_path', type=str, default='save',
                    help='directory to store checkpointed models')

    parser.add_argument('--id', type=str, default='',
                    help='an id identifying this run/job. used in cross-val and appended when writing progress files')
    #Optimization: for the Language Model
    parser.add_argument('--optim', type=str, default='adam',
                    help='what update to use? rmsprop|sgd|sgdmom|adagrad|adam')
    parser.add_argument('--learning_rate', type=float, default=4e-4,
                    help='learning rate')
    parser.add_argument('--learning_rate_decay_start', type=int, default=-1, 
                    help='at what iteration to start decaying learning rate? (-1 = dont) (in epoch)')
    parser.add_argument('--learning_rate_decay_every', type=int, default=10, 
                    help='every how many iterations thereafter to drop LR by half?(in epoch)')
    parser.add_argument('--optim_alpha', type=float, default=0.8,
                    help='alpha for adam')
    parser.add_argument('--optim_beta', type=float, default=0.999,
                    help='beta used for adam')
    parser.add_argument('--optim_epsilon', type=float, default=1e-8,
                    help='epsilon that goes into denominator for smoothing')

    parser.add_argument('--save_checkpoint_every', type=int, default=2500,
                    help='how often to save a model checkpoint (in iterations)?')
    parser.add_argument('--beam_size', type=int, default=3,
                    help='used when sample_max = 1, indicates number of beams in beam search. Usually 2 or 3 works well. More is not better. Set this to 1 for faster runtime but a bit worse performance.')

    parser.add_argument('--load_best_score', type=int, default=1,
                    help='Do we load previous best score when resuming training.')  
    parser.add_argument('--language_eval', type=int, default=1,
                    help='Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? requires coco-caption code from Github.')
  
    parser.add_argument('--losses_log_every', type=int, default=25,
                    help='How often do we snapshot losses, for inclusion in the progress dump? (0 = disable)')    

     # Optimization: General
    parser.add_argument('--max_epochs', type=int, default=30,
                    help='number of epochs')

    args = parser.parse_args()

    return args