import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
import skimage
import skimage.transform
import skimage.io



def crop_image(x, target_height=448, target_width=448):
    image = skimage.img_as_float(skimage.io.imread(x)).astype(np.float32)
    #return image

    if len(image.shape) == 2:
        image = np.tile(image[:,:,None], 3)
    elif len(image.shape) == 4:
        image = image[:,:,:,0]

    height, width, rgb = image.shape
    if width == height:
        resized_image = cv2.resize(image, (target_height,target_width))

    elif height < width:
        resized_image = cv2.resize(image, (int(width * float(target_height)/height), target_width))
        cropping_length = int((resized_image.shape[1] - target_height) / 2)
        resized_image = resized_image[:,cropping_length:resized_image.shape[1] - cropping_length]

    else:
        resized_image = cv2.resize(image, (target_height, int(height * float(target_width) / width)))
        cropping_length = int((resized_image.shape[0] - target_width) / 2)
        resized_image = resized_image[cropping_length:resized_image.shape[0] - cropping_length,:]

    return cv2.resize(resized_image,(target_height,target_width))


def visulize_attention(alphas,img_path,save_path):
    img=skimage.img_as_float(skimage.io.imread(img_path)).astype(np.float32)


    #cv2.resize(img,(224, 224))
    alphas=np.array(alphas).swapaxes(0,1) # n,49,1

    plt.imshow(img)
    plt.axis('off')

    #alpha_img=skimage.transform.pyramid_expand(alphas[0,:].reshape(7,7),upscale=16,sigma=20)
    alpha_img=skimage.transform.resize(alphas[0,:].reshape(14,14),[img.shape[0],img.shape[1]])
    plt.imshow(alpha_img,alpha=0.8)
    plt.set_cmap(cm.Greys_r)
    plt.axis('off')

    plt.savefig(save_path)
    plt.close()



