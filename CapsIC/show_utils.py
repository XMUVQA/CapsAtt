from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
import collections
import six
from functools import reduce
from operator import mul


from PIL import Image, ImageDraw
def visualize(image_file, proposals, att_weights, save_path):
    """
    :param image_file:  image_filename.jpg
    :param proposals:   36 * [scaled_x,
                              scaled_y,
                              scaled_x + scaled_width,
                              scaled_y + scaled_height,
                              scaled_width,
                              scaled_height]
    :param att_weights: [36, ]
    :param save_path:   save path
    :return:
    """

    def draw_box(proposal, mask):
        # draw a rectangle on the mask
        img_w, img_h = mask.size
        x1, y1 = proposal[0] * img_w, proposal[1] * img_h
        x2, y2 = proposal[2] * img_w, proposal[3] * img_h
        # w, h = proposal[4] * img_w, proposal[5] * img_h
        draw = ImageDraw.Draw(mask)
        draw.rectangle((x1, y1, x2, y2), fill=(255, 255, 255))
        return mask

    def draw_line(img, box1, box2, att, width_factor=1000):
        # draw a line between the box1's center point and the box2's center point
        img_w, img_h = img.size
        center1_x = int((box1[0] + box1[2]) * img_w / 2.0)
        center1_y = int((box1[1] + box1[3]) * img_h / 2.0)
        center2_x = int((box2[0] + box2[2]) * img_w / 2.0)
        center2_y = int((box2[1] + box2[3]) * img_h / 2.0)
        # print att
        att = int(att * width_factor)
        # print att
        draw = ImageDraw.Draw(img)
        draw.line([(center1_x, center1_y), (center2_x, center2_y)], fill=(255, 0, 0), width=att)

    assert proposals.shape[0] == att_weights.shape[0], 'length not match'
    img = Image.open(image_file)
    img = img.convert('RGBA')
    mask = Image.new("RGBA", img.size, (0, 0, 0, 0))

    for i in range(proposals.shape[0]):
        mask = draw_box(proposals[i], mask)
        img = Image.blend(img, mask, att_weights[i])
    # count = 0
    """
    att_sorted_idx = np.argsort(att_weights)
    att_weights = att_weights.reshape(len(att_weights), 1)
    Att = np.dot(att_weights, att_weights.T)
    for i in att_sorted_idx[-5:]:
        for j in att_sorted_idx[-5:]:
            draw_line(img, proposals[i], proposals[j], Att[i][j])
    """

    img.save(save_path)
    #img.show()
# # visualize(image_file, proposals, att_weights, './')
import os.path as osp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import skimage
import skimage.transform
import skimage.io
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

def att_visualize(image_file, proposals, att_weights, save_path):
#     """
#     :param image_file:  image filename
#     :param proposals:   R * [scaled_x,
#                              scaled_y,
#                              scaled_x + scaled_width,
#                              scaled_y + scaled_height,
#                              scaled_width,
#                              scaled_height]
#     :param att_weights: [R, ]
#     :param save_path:   save path
#     :return:
#     """

    def draw_box(proposal, att_weight):
        x1, y1 = proposal[0] * img_w, proposal[1] * img_h
        # x2, y2 = proposal[2] * img_w, proposal[3] * img_h
        w, h = proposal[4] * img_w, proposal[5] * img_h
        rect = plt.Rectangle((x1, y1), w, h, facecolor='white', edgecolor=None, alpha=att_weight*5)
        plt.gca().add_patch(rect)
        return

    assert proposals.shape[0] == att_weights.shape[0], 'length not match'
    
    img = skimage.img_as_float(skimage.io.imread(image_file)).astype(np.float32)
    img_h, img_w = img.shape[0], img.shape[1]
    plt.imshow(img)
    plt.axis('off')
    plt.gca().add_patch(plt.Rectangle((0, 0), img_w, img_h, facecolor='black', edgecolor=None, alpha=0.5))
    for i in range(proposals.shape[0]):
       draw_box(proposals[i], att_weights[i])
    plt.savefig(save_path, format='jpg')
    # plt.show()
    plt.close()

def Visualize_Q(image_file, proposals, Q_alpha1, save_path):
    """
    :param image_file:  image path
    :param proposals:   R * [scaled_x,
                              scaled_y,
                              scaled_x + scaled_width,
                              scaled_y + scaled_height,
                              scaled_width,
                              scaled_height]
    :param Q_alpha:    [K, R]
    :param save_path:   save path
    :return:
    """
    def draw_rects_frcnn(proposal, facecolor, alpha):
        x, y = proposal[0] * img_w, proposal[1] * img_h
        w, h = proposal[4] * img_w, proposal[5] * img_h

        rect = plt.Rectangle((x, y), w, h,
                             facecolor=facecolor, edgecolor='black', alpha=alpha)
        plt.gca().add_patch(rect)
        return None

    # load the input image through PIL.Image
    img_path = osp.join(image_file)
    img = Image.open(img_path, mode='r')
    img_w, img_h = img.size[0], img.size[1]
    colors = ['red', 'green', 'blue', 'yellow', 'white', 'aqua', 'purple', 'pink', 'brown']
    Region = len(proposals)

    plt.subplots(nrows=1, ncols=1, figsize=(0.01 * img_w, 0.01 * img_h))
    plt.imshow(img)
    plt.axis('off')

    Q_alpha1 = Q_alpha1[:4,:].transpose([1,0])  # R,4
    # draw boxes
    # for proposal, col_idx in zip(proposals, np.argmax(Q_alpha1, 0)):
    #     draw_rects_frcnn(proposal, colors[col_idx], alpha=0.2)
    for i in range(Region):
        draw_rects_frcnn(proposals[i], colors[np.argmax(Q_alpha1[i])], alpha=0.25)
            
    # adjust the figure
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)

    # save or show figure
    plt.savefig(save_path, format='jpg')
    # plt.show()
    plt.close()

    return None

def Visualize_att(image_file, proposals, vg_att, save_path):
    """
    :param image_file:  image path
    :param proposals:   R * [scaled_x,
                              scaled_y,
                              scaled_x + scaled_width,
                              scaled_y + scaled_height,
                              scaled_width,
                              scaled_height]
    :param vg_att:    [R, ]
    :param save_path:   save path
    :return:
    """
    def draw_rects_frcnn(proposal, facecolor, alpha):
        x, y = proposal[0] * img_w, proposal[1] * img_h
        w, h = proposal[4] * img_w, proposal[5] * img_h

        rect = plt.Rectangle((x, y), w, h,
                             facecolor=facecolor, edgecolor='white',linewidth=2, alpha=alpha)
        plt.gca().add_patch(rect)
        return None

    # load the input image through PIL.Image
    img_path = osp.join(image_file)
    img = Image.open(img_path, mode='r')
    img_w, img_h = img.size[0], img.size[1]
    
    plt.subplots(nrows=1, ncols=1, figsize=(0.01 * img_w, 0.01 * img_h))
    plt.imshow(img)
    plt.axis('off')
    
    rect = plt.Rectangle((0, 0), img_w, img_h, facecolor='white', edgecolor='none', alpha=0.5)
    plt.gca().add_patch(rect)

    # draw boxes
    for proposal, att in zip(proposals, vg_att):   
        draw_rects_frcnn(proposal, 'red', alpha=att*0.7)


    # adjust the figure
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)

    # save or show figure
    plt.savefig(save_path, format='jpg')
    # plt.show()
    plt.close()
    return None


def get_dataloader(opt):
    if opt.cnn_model=="frcnn":
        import frcnn_dataloader 
        dl = frcnn_dataloader.DataLoader(opt)
    elif opt.cnn_model=="vgg16" or opt.cnn_model=="vgg19":
        import cnn_dataloader
        dl = cnn_dataloader.DataLoader(opt)
    else:
        import resnet_dataloader
        dl = resnet_dataloader.DataLoader(opt)
    return dl


def get_num_params():
    num_params = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        num_params += reduce(mul, [dim.value for dim in shape], 1)
    return num_params
def idtransword(captions,id2word):
    
    #print r,c
    result = []
    #print captions
    for i in range(len(captions)):
        tmp = ""
        for j in range(len(captions[i])):
            if captions[i][j] == 0:
                break
            else:
                tmp += id2word[captions[i][j]] + " "
        result.append(tmp)
    return result

def idtransword2(captions,id2word):
    r,c = captions.shape
    #print r,c
    result = []
    #print captions
    for i in range(r):
        tmp = ""
        for j in range(c):
            if captions[i][j] == 0:
                break
            else:
                tmp += id2word[captions[i][j]] + " "
        result.append(tmp)
    return result

# My own clip by value which could input a list of tensors
def clip_by_value(t_list, clip_value_min, clip_value_max, name=None):
    if (not isinstance(t_list, collections.Sequence)
            or isinstance(t_list, six.string_types)):
        raise TypeError("t_list should be a sequence")
    t_list = list(t_list)
        
    with tf.name_scope(name or "clip_by_value") as name:
        values = [
            tf.convert_to_tensor(
                t.values if isinstance(t, tf.IndexedSlices) else t,
                name="t_%d" % i)
            if t is not None else t
            for i, t in enumerate(t_list)]
        values_clipped = []
        for i, v in enumerate(values):
            if v is None:
                values_clipped.append(None)
            else:
                with tf.get_default_graph().colocate_with(v):
                    values_clipped.append(
                        tf.clip_by_value(v, clip_value_min, clip_value_max))

        list_clipped = [
            tf.IndexedSlices(c_v, t.indices, t.dense_shape)
            if isinstance(t, tf.IndexedSlices)
            else c_v
            for (c_v, t) in zip(values_clipped, t_list)]

    return list_clipped

# Truncate the list of beam given a maximum length
def truncate_list(l, max_len):
    if max_len == -1:
        max_len = len(l)
    return l[:min(len(l),  max_len)]

# Turn nested state into a flattened list
# Used both for flattening the nested placeholder states and for output states value of previous time step
def flatten_state(state):
    if isinstance(state, tf.contrib.rnn.LSTMStateTuple):
        return [state.c, state.h]
    elif isinstance(state, tuple):
        result = []
        for i in xrange(len(state)):
            result += flatten_state(state[i])
        return result
    else:
        return [state]

# When decoding step by step: we need to initialize the state of next timestep according to the previous time step.
# Because states could be nested tuples or lists, so we get the states recursively.
def get_placeholder_state(state_size, scope = 'placeholder_state'):
    with tf.variable_scope(scope):
        if isinstance(state_size, tf.contrib.rnn.LSTMStateTuple):
            c = tf.placeholder(tf.float32, [None, state_size.c], name='LSTM_c')
            h = tf.placeholder(tf.float32, [None, state_size.h], name='LSTM_h')
            return tf.contrib.rnn.LSTMStateTuple(c,h)
        elif isinstance(state_size, tuple):
            result = [get_placeholder_state(state_size[i], "layer_"+str(i)) for i in xrange(len(state_size))]
            return tuple(result)
        elif isinstance(state_size, int):
            return tf.placeholder(tf.float32, [None, state_size], name='state')

# Get the last hidden vector. (The hidden vector of the deepest layer)
# For the input of the attention model of next time step.
def last_hidden_vec(state):
    if isinstance(state, tuple):
        return last_hidden_vec(state[len(state) - 1])
    elif isinstance(state, tf.contrib.rnn.LSTMStateTuple):
        return state.h
    else:
        return state

# Input: seq, N*D numpy array, with element 0 .. vocab_size. 0 is END token.
def decode_sequence(ix_to_word, seq):
    N, D = seq.shape
    out = []
    for i in range(N):
        txt = ''
        for j in range(D):
            ix = seq[i,j]
            if ix > 0 :
                if j >= 1:
                    txt = txt + ' '
                txt = txt + ix_to_word[str(ix)]
            else:
                break
        out.append(txt)
    return out

def get_initial_state(input, state_size, scope = 'init_state'):
    """
    Recursively initialize the first state.
    state_size is a nested of tuple and LSTMStateTuple and integer.
        
    It is so complicated because we use state_is_tuple
    """

    with tf.variable_scope(scope):
        if isinstance(state_size, tf.contrib.rnn.LSTMStateTuple):
            c = slim.fully_connected(input, state_size.c, activation_fn=tf.nn.tanh, scope='LSTM_c')
            h = slim.fully_connected(input, state_size.h, activation_fn=tf.nn.tanh, scope='LSTM_h')
            return tf.contrib.rnn.LSTMStateTuple(c,h)
        elif isinstance(state_size, tuple):
            result = [get_initial_state(input, state_size[i], "layer_"+str(i)) for i in xrange(len(state_size))]
            return tuple(result)
        elif isinstance(state_size, int):
            return slim.fully_connected(input, state_size, activation_fn=tf.nn.tanh, scope='state')

def expand_feat(input, multiples, scope = 'expand_feat'):
    """
    Expand the dimension of states;
    According to multiples.
    Similar reason why it's so complicated.
    """
    with tf.variable_scope(scope):
        if isinstance(input, tf.contrib.rnn.LSTMStateTuple):
            c = expand_feat(input.c, multiples, scope='expand_LSTM_c')
            h = expand_feat(input.h, multiples, scope='expand_LSTM_c')
            return tf.contrib.rnn.LSTMStateTuple(c,h)
        elif isinstance(input, tuple):
            result = [expand_feat(input[i], multiples, "expand_layer_"+str(i)) for i in xrange(len(input))]
            return tuple(result)
        else:
            return tf.reshape(tf.tile(tf.expand_dims(input, 1), [1, multiples, 1]), [tf.shape(input)[0] * multiples, input.get_shape()[1].value])

def get_optimizer(opt, lr):
    if opt.optim == 'rmsprop':
        return tf.train.RMSPropOptimizer(lr, momentum=opt.optim_alpha, epsilon=opt.optim_epsilon)
    elif opt.optim == 'adagrad':
        return tf.train.AdagradOptimizer(lr)
    elif opt.optim == 'sgd':
        return tf.train.GradientDescentOptimizer(lr)
    elif opt.optim == 'sgdm':
        return tf.train.MomentumOptimizer(lr, opt.optim_alpha)
    elif opt.optim == 'sgdmom':
        return tf.train.MomentumOptimizer(lr, opt.optim_alpha, use_nesterov=True)
    elif opt.optim == 'adam':
        return tf.train.AdamOptimizer(lr, beta1=opt.optim_alpha, beta2=opt.optim_beta, epsilon=opt.optim_epsilon)
    else:
        raise Exception('bad option opt.optim')

def get_cnn_optimizer(opt, cnn_lr):
    if opt.cnn_optim == 'rmsprop':
        return tf.train.RMSPropOptimizer(cnn_lr, momentum=opt.cnn_optim_alpha, epsilon=opt.optim_epsilon)
    elif opt.cnn_optim == 'adagrad':
        return tf.train.AdagradOptimizer(cnn_lr)
    elif opt.cnn_optim == 'sgd':
        return tf.train.GradientDescentOptimizer(cnn_lr)
    elif opt.cnn_optim == 'sgdm':
        return tf.train.MomentumOptimizer(cnn_lr, opt.cnn_optim_alpha)
    elif opt.cnn_optim == 'sgdmom':
        return tf.train.MomentumOptimizer(cnn_lr, opt.cnn_optim_alpha, use_nesterov=True)
    elif opt.cnn_optim == 'adam':
        return tf.train.AdamOptimizer(cnn_lr, beta1=opt.cnn_optim_alpha, beta2=opt.cnn_optim_beta, epsilon=opt.optim_epsilon)
    else:
        raise Exception('bad option opt.cnn_optim')