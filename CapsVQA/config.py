__author__ = 'antony'

# what data to use for training
TRAIN_DATA_SPLITS = 'train+val'
# what data to use for the vocabulary
QUESTION_VOCAB_SPACE = 'train+val'
ANSWER_VOCAB_SPACE = 'train+val' # test/test-dev/genome should not appear here
# vqa tools - get from https://github.com/VT-vision-lab/VQA
VQA_TOOLS_PATH = '../VQA/CODE/PythonHelperTools'
VQA_EVAL_TOOLS_PATH = '/VQA/CODE/PythonEvaluationTools'
# location of the data
VQA_PREFIX = '' # VQA dataset files
FEATURE_ROOT='' # The root for extracted feature files.
DATA_PATHS = {
	'train': {
		'ques_file': VQA_PREFIX + '/v2_OpenEnded_mscoco_train2014_questions.json',
		'ans_file': VQA_PREFIX + '/v2_mscoco_train2014_annotations.json',
		'features_prefix': FEATURE_ROOT + 'train/'
	},
	'val': {
		'ques_file': VQA_PREFIX + '/v2_OpenEnded_mscoco_val2014_questions.json',
		'ans_file': VQA_PREFIX + '/v2_mscoco_val2014_annotations.json',
		'features_prefix': FEATURE_ROOT + 'val/'
	},
	'test-dev': {
		'ques_file': VQA_PREFIX + '/v2_OpenEnded_mscoco_test-dev2015_questions.json',
		'features_prefix': FEATURE_ROOT + 'test/'
	},
	'test': {
		'ques_file': VQA_PREFIX + '/v2_OpenEnded_mscoco_test2015_questions.json',
		'features_prefix': FEATURE_ROOT + 'test/'
	},
	'genome': {
		'genome_file': VQA_PREFIX + '/Questions/OpenEnded_genome_train_questions.json',
		'features_prefix': FEATURE_ROOT + 'vg/'
	}
}
# Model Setting
ITER_NUM=3
CNN_DIM=2048
GLOVE_DIM = 300
ANSWER_DIM = 3000      #the top 3000 answers
QUESTION_MAX_LENGTH = 15
WEIGHT_DECAY=0.00005
QUESTION_RNN_DIM = 1024
FF_DIM=2048
OUT_FF_DIM=2048
CNN_REGION_NUM = 37 # 36 FRCNN features and 1 fake region
ATTENT_DIM = 1024
# Experiment Setting
KEEP_PROB = 0.8
LEARNING_RATE = 0.0007
VAL_STEP=10000
MAX_TRAIN_STEP=200000
SAVE_STEP=10000
BATCH_SIZE=128
VAL_BATCH_SIZE=128
SHOW_STEP=100
DECAY_STEP=25000
DECAY_RATE=0.5
MODEL_SCOPE='CapsAtt'
TEST_MODE='test' # or "test-dev" for VQA1.0

RESTORE_NUM=0
MODEL_NAME='caps_frcnn_s%d'%ITER_NUM
MODEL_LOAD_NAME='%s-%d'%(MODEL_NAME,RESTORE_NUM)
USE_PRE_VQA_MODEL=False 
FOLDER_NAME='vqa2_caps_s%d_%d_A%d_%s_DEC%d'%(ITER_NUM,BATCH_SIZE,ATTENT_DIM,TRAIN_DATA_SPLITS,DECAY_STEP)
EVALUATION_SAVE_PATH=FOLDER_NAME+'%s_results.json'%MODEL_NAME
GPU_ID='2'