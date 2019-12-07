import os 
import opts
from bottomup_CapAtt import *

import time
import eval_utils   
from six.moves import cPickle
import show_utils as utils

def train(opt):
    dl = utils.get_dataloader(opt)
    opt.vocab_size = 10010
    model = bottomCapAttV3(opt)
    infos = {}
    iteration = infos.get('iter', 0)
    epoch = infos.get('epoch', 0)
    val_result_history = infos.get('val_result_history', {})
    loss_history = infos.get('loss_history', {})
    if opt.load_best_score == 1:
        best_val_score = infos.get('best_val_score', None)

    with tf.variable_scope(tf.get_variable_scope()):
        model.build_model()
        model.build_generator()
        model.build_decoder()
        


    config = tf.ConfigProto(allow_soft_placement = True)
    #config.gpu_options.per_process_gpu_memory_fraction=0.9
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        model.initialize(sess)
        sess.run(tf.assign(model.lr, opt.learning_rate))   
        learning_rate =opt.learning_rate
        sess.run(tf.assign(model.training, True))           
        start = time.time()
        while True:

            #print 'read data:'+str(time.time()-start)
            data = dl.get_batch(split="train",batch_size=opt.batch_size)
            batch_size=data['labels'].shape[0]
            feed = {model.labels: data['labels'],
                        model.images:data['features'],
                        model.masks:data['masks'],
                        model.b:np.zeros([batch_size,opt.num_boxes,1,1],dtype=float)

                    }
            
            train_loss, merged, _,lr = sess.run([model.cost,model.summaries,model.train_op,model.lr],feed)

            iteration += 1
            
            if data['wrapped']:
                epoch += 1
                """
                if epoch > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0:
                    frac = float(epoch - opt.learning_rate_decay_start) / opt.learning_rate_decay_every
                    decay_factor = 0.5  ** frac
                    sess.run(tf.assign(model.lr, opt.learning_rate * decay_factor)) # set the decayed rate     
                else:
                    sess.run(tf.assign(model.lr, opt.learning_rate))  
                """
                if opt.learning_rate_decay_start>=0:
                    if epoch%2==0:
                        learning_rate =learning_rate/2
                        sess.run(tf.assign(model.lr, learning_rate))      
            # Write the training loss summary
            if (iteration % opt.losses_log_every == 0):
                model.summary_writer.add_summary(merged, iteration)
                model.summary_writer.flush()
                loss_history[iteration] = train_loss


            if iteration%100==0:
                end = time.time()
                print "iter {} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f},lr={:.6f}" \
                        .format(iteration, epoch, train_loss, end - start, lr)
                start = time.time()
            if iteration%1000==0:
                batch_size=data['features'].shape[0]
                feed = { 
                    model.images:data['features'],
                    model.b:np.zeros([batch_size,opt.num_boxes,1,1],dtype=float)
                }
                captions = sess.run(model.generator,feed)
                print utils.idtransword(data['labels'][:5,1:],dl.id2word)
                print utils.idtransword(captions[:1,:],dl.id2word)
                
            if  iteration % opt.save_checkpoint_every == 0:
                val_loss, predictions, lang_stats = eval(sess,model,dl,opt)
                summary = tf.Summary(value=[tf.Summary.Value(tag='validation loss', simple_value=val_loss)])
                model.summary_writer.add_summary(summary, iteration)
                for k,v in lang_stats.iteritems():
                    summary = tf.Summary(value=[tf.Summary.Value(tag=k, simple_value=v)])
                    model.summary_writer.add_summary(summary, iteration)
                model.summary_writer.flush()
                val_result_history[iteration] = {'loss': val_loss, 'lang_stats': lang_stats, 'predictions': predictions}
                 
                if opt.language_eval == 1:
                    current_score = lang_stats['CIDEr']
                else:
                    current_score = - val_loss
           

                if best_val_score is None or current_score > best_val_score: # if true
                    best_val_score = current_score
                    checkpoint_path = os.path.join(opt.checkpoint_path, 'model.ckpt')
                    model.saver.save(sess, checkpoint_path, global_step = iteration)
                    print "**********************"
                    print "model saved to {}".format(checkpoint_path)
                    print "**********************"
                    # Dump miscalleous informations
                    infos['iter'] = iteration
                    infos['epoch'] = epoch
                    #infos['iterators'] = loader.iterators
                    infos['best_val_score'] = best_val_score
                    infos['opt'] = opt
                    infos['val_result_history'] = val_result_history
                    infos['loss_history'] = loss_history
                    
                    #infos['vocab'] = loader.get_vocab()
                    with open(os.path.join(opt.checkpoint_path, 'infos_'+opt.id+'.pkl'), 'wb') as f:
                        cPickle.dump(infos, f)

            # Stop if reaching max epochs
            if epoch >= opt.max_epochs and opt.max_epochs != -1:
                break
def eval(sess, model, loader, opt, language_eval=1):
    n = 0 
    loss_sum = 0
    loss_evals = 0
    predictions = []
    sess.run(tf.assign(model.training, False))
    start = time.time()
    
    training = sess.run(model.training)
    print "Eval start,training:{}".format(training)
    while True:
        data =  loader.get_batch(split="val",batch_size=10)
        batch_size=data['labels'].shape[0]
        n = n + 10
        feed = {
                model.labels: data['labels'],
                model.images:data['features'],
                model.masks:data['masks'],
                model.b:np.zeros([batch_size,opt.num_boxes,1,1],dtype=float)
            }

        loss = sess.run(model.cost,feed)
        loss_sum += loss
        loss_evals += 1
        
        
        batch_size=data['features'].shape[0]
        feed = { 
                model.images:data['features'],
                model.b:np.zeros([batch_size,opt.num_boxes,1,1],dtype=float)
            }
        seq = sess.run(model.generator,feed)
        captions = utils.idtransword(seq,loader.id2word)

        for i in range(len(data['imgid'])):
            imgid = data['imgid'][i]
            caption = captions[i]
            
            predictions.append({'image_id': imgid, 'caption': caption})

        
        if n%1000==0:
             print str(n),'/',str(5000)
        if data['wrapped']:
            break

        if  n>=5000:
            break

    #print predictions
    print 'eval cost:',str(time.time()-start),'s'
    if language_eval == 1:
        lang_stats = eval_utils.language_eval(predictions)
        #print lang_stats
    sess.run(tf.assign(model.training, True))
    return loss_sum/loss_evals,predictions,lang_stats


def eval_beamsearch(opt):
    dl = utils.get_dataloader(opt)
    opt.vocab_size = 10010
    model = bottomCapAttV3(opt)

    with tf.variable_scope(tf.get_variable_scope()):
        model.build_model()
        generated_captions = model.build_generator()
        model.build_decoder()

    config = tf.ConfigProto(allow_soft_placement = True)
    #config.gpu_options.per_process_gpu_memory_fraction=0.9
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        model.initialize(sess)
        eval(sess,model,dl,opt)
        sess.run(tf.assign(model.training, False))
        n = 0
        predictions = []
        while True:
            data =  dl.get_batch(split="val",batch_size=1)
            #print data['features'].shape
            n = n + 1
            seq = model.decode(data['features'], opt.beam_size, sess)
            captions = utils.idtransword(seq,dl.id2word)
            imgid = data['imgid'][0]
            caption = captions[0]
            
            predictions.append({'image_id': imgid, 'caption': caption})
            if n%1000==0:
                print str(n),'/',str(5000)
            if data['wrapped']:
                break
            
            if  n>=5000:
                break
        
        lang_stats = eval_utils.language_eval(predictions)
        
            
def viz(opt):
    opt.batch_size=1
    dl = utils.get_dataloader(opt)
    opt.vocab_size = 10010
    model = bottomCapAttV3(opt)
    with tf.variable_scope(tf.get_variable_scope()):
        model.build_model()
        model.build_generator()
    vis_dir = 'viz/our/'
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    config = tf.ConfigProto(allow_soft_placement = True)
    #config.gpu_options.per_process_gpu_memory_fraction=0.9
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        import numpy as np
        model.initialize(sess)
        sess.run(tf.assign(model.training, False))
        for i in range(10):
            data =  dl.get_batch(split="val",batch_size=1)
            feed = { 
                    model.images:data['features'],
                    model.b:np.zeros([1,opt.num_boxes,1,1],dtype=float)
                }
            seq,alpha_list = sess.run([model.generator,model.alpha_list],feed)
            
            captions = utils.idtransword(seq,dl.id2word)
            print captions
            print utils.idtransword(data['labels'][:5,1:],dl.id2word)

                        
            imgid = data['imgid']
            imgfile = "/home/data/COCO/DATA/val2014/COCO_val2014_"+str(imgid[0]).zfill(12)+".jpg"
            spatialfile = "/home/extra_data/hdd2/cwq/frcnn_feature/36/COCO/spatial_info/train/"+str(imgid[0])+".npy"
            proposals = np.load(spatialfile)

            print '%s: %s' % (str(imgid[0]), captions[0])
            save_name = vis_dir+str(imgid[0])
            if not os.path.exists(save_name):
                os.makedirs(save_name)
            words = captions[0].split()
            
            for i in range(len(captions[0].split())):
                for j in range(len(alpha_list[0])):
                    savefile = save_name+"/iter_%s_%s_%s.jpg"%(str(j),str(i), words[i])
                    
                    print savefile
                    att_weights = alpha_list[i][j][0]
                    att_weights = att_weights.reshape([36])
                    #print att_weights.shape
                    #print att_weights
                    utils.att_visualize(imgfile,proposals,att_weights,savefile)
                with open(save_name+"/caption.txt", 'w') as f:
                    f.write(captions[0])

        
def test_with_beamsearch(opt):
    dl = utils.get_dataloader(opt)
    opt.vocab_size = 10010
    model = bottomCapAttV3(opt)


    with tf.variable_scope(tf.get_variable_scope()):
        model.build_model()
        model.build_generator()
        model.build_decoder()

    config = tf.ConfigProto(allow_soft_placement = True)
    #config.gpu_options.per_process_gpu_memory_fraction=0.9
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        model.initialize(sess)
        eval(sess,model,dl,opt)
        sess.run(tf.assign(model.training, False))
        n = 0
        predictions = []
        print "ready test!!!"
        
        while True:
 
            data =  dl.get_batch(split="val",batch_size=10)
            n = n + 10
            batch_size=data['features'].shape[0]
            feed = { 
                       model.images:data['features'],
                       model.b:np.zeros([1,opt.num_boxes,1,1],dtype=float)
            }
            seq = sess.run(model.generator,feed)
            captions = utils.idtransword(seq,dl.id2word)
            for i in range(len(data['imgid'])):
                imgid = data['imgid'][i]
                caption = captions[i]
                    
                predictions.append({'image_id': imgid, 'caption': caption})
            #print len(predictions)
            if n%1000==0:
                print str(n),'/',str(5000)
            if data['wrapped']:
                print "dataset over!!!"
                break
            
            if  n>=5000:
                break
        print len(predictions)
        lang_stats = eval_utils.language_eval(predictions)
        
        predictions=[]
        n=0
        while True:
            data =  dl.get_test_batch(batch_size=1)        
            n = n + 1
                
            seq = model.decode(data['features'], opt.beam_size,sess)
            captions = utils.idtransword(seq,dl.id2word)
            imgid = data['imgid'][0]
            caption = captions[0]
                
            predictions.append({'image_id': imgid, 'caption': caption})
            if n%1000==0:
                print str(n),'/',str(5000)
            if data['wrapped']:
                print n
                print "dataset over!!!"
                break
            
            if  n>=5000:
                break
        print len(predictions)
        lang_stats = eval_utils.language_eval(predictions)

opt = opts.parse_opt()
if opt.eval_or_train=="train":
    train(opt)
elif opt.eval_or_train=="eval":
    eval_beamsearch(opt)
elif opt.eval_or_train=="viz":
    viz(opt)
elif opt.eval_or_train=="test":
    test_with_beamsearch(opt)
            
            
           