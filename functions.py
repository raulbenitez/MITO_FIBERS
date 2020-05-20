import os
import read_lif
from PIL import Image
from pathlib import Path
import cv2
import numpy as np
import tensorflow as tf
import skimage.transform
import skimage.io as io
import warnings
import pandas as pd
warnings.filterwarnings('ignore')


def readLIF(filepath, maxZ=False):
    reader = read_lif.Reader(filepath)
    series = reader.getSeries()
    r = series[0].getFrame(T=0, channel=0)  # image is a numpy array, first time point & first channel
    b = series[0].getFrame(T=0, channel=2)
    if maxZ==False:
        r1 = np.zeros([r.shape[0],r.shape[1]])
        b1 = np.zeros([b.shape[0],b.shape[1]])
        for i in range(8):
            r1 += r[:,:,i]
            b1 += b[:,:,i]
        img = skimage.transform.rotate(np.stack([r1,b1],axis=2)/(8*255),270)
        img = cv2.flip(img,1)
    else:
        meanR = np.zeros(8)
        for i in range(8):
            meanR[i] = np.mean(r[:,:,i])
        maxR = np.argmax(meanR)
        r1 = r[:,:,maxR]
        b1 = b[:,:,maxR]
        img = skimage.transform.rotate(np.stack([r1,b1],axis=2)/(255),270)
        img = cv2.flip(img,1)
        
    return(img[:,:,0],img[:,:,1])


def crop_img(img):
    im = np.zeros((256,256,16))
    im[:,:,0] = img[0:256,0:256]
    im[:,:,1] = img[256:512,0:256]
    im[:,:,2] = img[512:768,0:256]
    im[:,:,3] = img[768:1024,0:256]
    im[:,:,4] = img[0:256,256:512]
    im[:,:,5] = img[256:512,256:512]
    im[:,:,6] = img[512:768,256:512]
    im[:,:,7] = img[768:1024,256:512]
    im[:,:,8] = img[0:256,512:768]
    im[:,:,9] = img[256:512,512:768]
    im[:,:,10] = img[512:768,512:768]
    im[:,:,11] = img[768:1024,512:768]
    im[:,:,12] = img[0:256,768:1024]
    im[:,:,13] = img[256:512,768:1024]
    im[:,:,14] = img[512:768,768:1024]
    im[:,:,15] = img[768:1024,768:1024]
    return im

def crop_img_rgb(img):
    im = np.zeros((256,256,3,16))
    im[:,:,:,0] = img[0:256,0:256,:]
    im[:,:,:,1] = img[256:512,0:256,:]
    im[:,:,:,2] = img[512:768,0:256,:]
    im[:,:,:,3] = img[768:1024,0:256,:]
    im[:,:,:,4] = img[0:256,256:512,:]
    im[:,:,:,5] = img[256:512,256:512,:]
    im[:,:,:,6] = img[512:768,256:512,:]
    im[:,:,:,7] = img[768:1024,256:512,:]
    im[:,:,:,8] = img[0:256,512:768,:]
    im[:,:,:,9] = img[256:512,512:768,:]
    im[:,:,:,10] = img[512:768,512:768,:]
    im[:,:,:,11] = img[768:1024,512:768,:]
    im[:,:,:,12] = img[0:256,768:1024,:]
    im[:,:,:,13] = img[256:512,768:1024,:]
    im[:,:,:,14] = img[512:768,768:1024,:]
    im[:,:,:,15] = img[768:1024,768:1024,:]
    return im

def save_crops(img,filename,savefolder,subfolder):
    savepath = os.path.join(savefolder,subfolder)
    try:
        os.makedirs(savepath)
    except:
        pass
    if img.ndim ==  3:
        for i in range(img.shape[2]):
            img1 = Image.fromarray(np.uint8(img[:,:,i]*255),'L')
            img1.save(os.path.join(savepath,filename[:-4]+'_'+str(i).zfill(2)+'.png'))
    elif img.ndim == 4:
        for i in range(img.shape[3]):
            img1 = Image.fromarray(np.uint8(img[:,:,:,i]*255),'RGB')
            img1.save(os.path.join(savepath,filename[:-4]+'_'+str(i).zfill(2)+'.png'))
    else:
        print('Error saving files')
   
              
        
def savePNG(path,crop=0):
    lif_path = os.path.join(path,'Lif')
    png_path = os.path.join(path,'png')
    try:
        os.makedirs(os.path.join(png_path,'red'))
        os.makedirs(os.path.join(png_path,'blue'))
        os.makedirs(os.path.join(png_path,'rbb'))
    except:
        pass
    files_lif = sorted([f.name for f in os.scandir(lif_path)])
    for i in range(len(files_lif)):
        filepath = os.path.join(lif_path,files_lif[i])
        (R, B) = readLIF(filepath)
        RBB = np.stack([R,B,B],axis=2)
        R1 = Image.fromarray(np.uint8(R*255),'L')
        B1 = Image.fromarray(np.uint8(B*255),'L')
        RBB1 = Image.fromarray(np.uint8(RBB*255),'RGB')
        R1.save(os.path.join(png_path,'red',files_lif[i][:-4]+'.png'))
        B1.save(os.path.join(png_path,'blue',files_lif[i][:-4]+'.png'))
        RBB1.save(os.path.join(png_path,'rbb',files_lif[i][:-4]+'.png'))
        if crop==1:
            savefolder = os.path.join(path,'Crops',files_lif[i][:-4])
            try:
                os.makedirs(os.path.join(savefolder,'red'))
                os.makedirs(os.path.join(savefolder,'blue'))
                os.makedirs(os.path.join(savefolder,'rbb'))
            except:
                pass
            im_R = crop_img(R)
            im_B = crop_img(B)
            im_RBB = crop_img_rgb(RBB)
            save_crops(im_R,files_lif[i],savefolder,'red')
            save_crops(im_B,files_lif[i],savefolder,'blue')
            save_crops(im_RBB,files_lif[i],savefolder,'rbb')    
    

def trainGenerator(batch_size,train_path,image_folder_R,image_folder_B,mask_folder,aug_dict,image_color_mode = "grayscale",mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",save_to_dir = None,target_size = (256,256),seed = 1):
    image_datagen_R = tf.keras.preprocessing.image.ImageDataGenerator(**aug_dict)
    image_datagen_B = tf.keras.preprocessing.image.ImageDataGenerator(**aug_dict)
    mask_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**aug_dict)
    image_generator_R = image_datagen_R.flow_from_directory(train_path,classes = [image_folder_R],class_mode = None,color_mode = image_color_mode,target_size = target_size,batch_size = batch_size,save_to_dir = save_to_dir,save_prefix  = image_save_prefix,seed = seed)
    image_generator_B = image_datagen_B.flow_from_directory(train_path,classes = [image_folder_B],class_mode = None,color_mode = image_color_mode,target_size = target_size,batch_size = batch_size,save_to_dir = save_to_dir,save_prefix  = image_save_prefix,seed = seed)
    mask_generator = mask_datagen.flow_from_directory(train_path,classes = [mask_folder],class_mode = None,color_mode = mask_color_mode,target_size = target_size,batch_size = batch_size,save_to_dir = save_to_dir,save_prefix  = mask_save_prefix,seed = seed)
    train_generator = zip(image_generator_R,image_generator_B,mask_generator)
    for (img_R,img_B,mask) in train_generator:
        img,mask = adjustData(img_R,img_B,mask)
        yield (img,mask)

        
def adjustData(img_R,img_B,mask):
    if(np.max(img_R) > 1):
        img_R = img_R / 255
        img_B = img_B / 255
        img = np.concatenate((img_R,img_B), axis = 3)
        mask = mask /255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    else:
        img = np.concatenate((img_R,img_B), axis = 3)
    return (img,mask)


def testGenerator(testfolder,files,target_size = (256,256,2),as_gray = True):    
    test_path_R = os.path.join(testfolder,"image_r")
    test_path_B = os.path.join(testfolder,"image_b")
    files = [f.name for f in os.scandir(test_path_R) if f.is_file and f.name.endswith('.png')]
    for i in range(len(files)):
        img_R = io.imread(os.path.join(test_path_R,files[i]),as_gray = as_gray)
        img_R = img_R / 255
        img_B = io.imread(os.path.join(test_path_B,files[i]),as_gray = as_gray)
        img_B = img_B / 255
        img = np.stack((img_R,img_B), axis = 2)
        img = skimage.transform.resize(img,target_size)
        img = np.reshape(img,(1,)+img.shape)
        yield img
        
        
def testGenerator2(crops_red,crops_blue):
    for i in range(16):
        img_R = crops_red[:,:,i] / 255
        img_B = crops_blue[:,:,i] / 255
        img = np.stack((img_R,img_B), axis = 2)
        img = np.reshape(img,(1,)+img.shape)
        yield img
    

def saveResults(save_path,npyfile,files):
    for i,item in enumerate(npyfile):
        img = item[:,:,0]
        img = (img*255).astype('uint8')
        io.imsave(os.path.join(save_path,files[i]),img)
        
        


