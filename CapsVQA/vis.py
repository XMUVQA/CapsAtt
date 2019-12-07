from  Solver import Solver
import dataloader
from Model import  CapsVQA
import tensorflow as tf
import config,os
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']=config.GPU_ID
def main():
    dataset=dataloader.prepare_test_dataset()
    with tf.variable_scope(config.VQA_SCOPE):
        model=CapsVQA(mode='test')
        model.build_vqa_module()
    solver=Solver(model,dataset)
    solver.visulize_att_lists()

if __name__ == '__main__':
	main()