import numpy as np
import os
from collections import defaultdict,Counter
import random
path = "../Up-Down-Captioner/data/coco_splits/"
class DataLoader():
    def __init__(self,opt):
        self.opt = opt
        self.caption_txt = [path+"train_captions.txt",path+"val_captions.txt"]
        self.vocabtxt = path + "train_vocab.txt"
        self.trainTxt = path + "karpathy_train_images.txt"
        self.valTxt = path + "karpathy_val_images.txt"
        self.testTxt = path + "karpathy_test_images.txt"
        self.num_boxes = opt.num_boxes
    
        self.feature_path = "/home/extra_data/hdd2/cwq/resnet_feat/resnet_res5c_bgrms_large/"
        #self.avg_feat_path = "data/coco/avg_frcnn_feature/"
        
        self.seq_per_img = self.opt.seq_per_img
        self.seq_length = self.opt.seq_length
        self.split_ix = {'train': [], 'val': [], 'test': []}
        
        self.getImgID('train',self.trainTxt)
        self.getImgID('val',self.valTxt)
        self.getImgID('test',self.testTxt)

        print 'assigned %d images to split train' %len(self.split_ix['train'])
        print 'assigned %d images to split val' %len(self.split_ix['val'])
        print 'assigned %d images to split test' %len(self.split_ix['test'])
        #print self.split_ix['train']
        self.data = {'train': [], 'val': [], 'test': []}


        self.load_captions(["train","val"])

        self.iterators = {'train': 0, 'val': 0, 'test': 0}
        self.word2id,self.id2word = self.getVocab(self.vocabtxt)


        #print self.split_ix['train']
        #pass
    def getVocab(self,filename):
        data = open(filename).readlines()
        word2id = {}
        id2word = {}
        count = 0
        for word in data:
            word = word[:-1]
            word2id[word] = count
            id2word[count] = word
            count+=1
        return word2id,id2word



    def random_shuffle(self,split="train"):
        random.shuffle(self.data[split])

    def getImgID(self,split,txtfile):
        for i in open(txtfile).readlines():
            imgid = int(i.split()[-1])
            #print imgid
            self.split_ix[split].append(imgid)

    def load_captions(self,splits):
        self.id_to_caption = defaultdict(list)
        for src in self.caption_txt:
            print 'Loading captions from: %s' % src
            with open(src) as txtfile:
                for line in txtfile.readlines():
                    image_id = int(line.split('.jpg')[0].split('_')[-1])
                    seq = [int(w) for w in line.split(' | ')[-1].split()]
                    if len(seq)<self.seq_length:
                        zfill = self.seq_length-len(seq)
                        for i in range(zfill):
                            seq.append(0)
                    else:
                        seq=seq[:self.seq_length]
                    if len(self.id_to_caption[image_id])<5:
                        self.id_to_caption[image_id].append(seq)
            print 'Loaded %d image ids' % len(self.id_to_caption)
        for split in splits:
            for imgid in self.split_ix[split]:
                captions = self.id_to_caption[imgid]
                assert len(captions)==5
                item = {}
                item['imgid'] = imgid
                item['captions'] = captions
                if split == 'train':
                    featFileExist = False
                    if os.path.exists(self.feature_path+"train2014/COCO_train2014_"+str(imgid).zfill(12)+".jpg.npz"):
                        featPath = self.feature_path+"train2014/COCO_train2014_"+str(imgid).zfill(12)+".jpg.npz"
                        featFileExist = True
                         
                    elif os.path.exists(self.feature_path+"val2014/COCO_val2014_"+str(imgid).zfill(12)+".jpg.npz"):
                        featPath = self.feature_path+"val2014/COCO_val2014_"+str(imgid).zfill(12)+".jpg.npz"
                        
                        featFileExist = True
                        
                    assert featFileExist==True
                else:
                    featFileExist = False
                    if os.path.exists(self.feature_path+"val2014/COCO_val2014_"+str(imgid).zfill(12)+".jpg.npz"):
                        featPath = self.feature_path+"val2014/COCO_val2014_"+str(imgid).zfill(12)+".jpg.npz"
                        
                        featFileExist = True
                    assert featFileExist==True
                item['feature_path'] = featPath
                #item['avg_feat_path'] = avgPath
                self.data[split].append(item)
            print len( self.data[split])

    def get_batch(self,split,batch_size=None):
        batch_size = batch_size or self.batch_size
        imgid = []
        label_batch = np.zeros([batch_size*self.seq_per_img,self.seq_length+2],dtype='int')
        mask_batch = np.zeros([batch_size*self.seq_per_img,self.seq_length+2],dtype='float')
        feature_batch = np.zeros([batch_size,self.num_boxes,2048],dtype='float32')
        #avg_feat_batch = np.zeros([batch_size*self.seq_per_img,2048],dtype='float32')

        wrapped = False
        max_index = len(self.data[split])
        

        ###
        """
        image
        info[id,file_path]

        """
        ###
        for i in range(batch_size):
            ri = self.iterators[split]
            
            ri_next = ri+1
            
            item = self.data[split][ri]
            imgid.append(item['imgid'])
            tmp = np.load(item['feature_path'])
            feat = tmp['x'].transpose(1,2,0)
            feat = np.reshape(feat,[-1,self.num_boxes,2048])
            feature_batch[i] = feat
            label_batch[i * self.seq_per_img : (i + 1) * self.seq_per_img, 1 : self.seq_length + 1] = item['captions']
            
            if ri_next >=max_index:
                ri_next=0
                wrapped = True
                self.random_shuffle(split=split)

            self.iterators[split] = ri_next

        nonzeros = np.array(map(lambda x: (x != 0).sum()+2, label_batch))
        for ix, row in enumerate(mask_batch):
            row[:nonzeros[ix]] = 1

        #print mask_batch
        
        data = {}
        data['imgid'] = imgid
        data['labels'] = label_batch
        data['masks'] = mask_batch
        data['wrapped'] = wrapped
        data['features'] = feature_batch
        #data['avg_feat'] = avg_feat_batch
        return data

    def get_test_batch(self,split='test',batch_size=None):
        batch_size = batch_size or self.batch_size
        imgids = []
        feature_batch = np.zeros([batch_size,self.num_boxes,2048],dtype='float32')
        wrapped = False
        max_index = len(self.split_ix[split])
        #print max_index
        for i in range(batch_size):
            ri = self.iterators[split]
            
            ri_next = ri+1
            imgid = self.split_ix[split][ri]
            imgids.append(imgid)
            feature_path = self.feature_path+"val2014/COCO_val2014_"+str(imgid).zfill(12)+".jpg.npz"
            tmp = np.load(feature_path)
            feat = tmp['x'].transpose(1,2,0)
            feat = np.reshape(feat,[-1,self.num_boxes,2048])
            feature_batch[i] = feat
            
            if ri_next >=max_index:
                ri_next=0
                wrapped = True
            self.iterators[split] = ri_next
        #print self.iterators[split]

        data = {}
        data['imgid'] = imgids
        data['wrapped'] = wrapped
        data['features'] = feature_batch

        return data


