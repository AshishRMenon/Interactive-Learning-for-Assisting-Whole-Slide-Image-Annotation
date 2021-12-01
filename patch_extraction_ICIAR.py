# -*- coding: utf-8 -*-
"""
ICIAR2018 - Grand Challenge on Breast Cancer Histology Images
https://iciar2018-challenge.grand-challenge.org/home/
"""
import openslide

import xml.etree.ElementTree as ET
import numpy as np
from scipy.misc import imsave
import cv2
import os
import argparse
from multiprocessing import Pool
import multiprocessing
import itertools
import argparse
import glob
from PIL import Image


def get_connected_components(img):
    img = cv2.cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]  # ensure binary
    num_labels, labels_im = cv2.connectedComponents(img)
    return num_labels,labels_im


def readXML(filename):
    
    tree = ET.parse(filename)
    
    root = tree.getroot()
    regions = root[0][1].findall('Region')
    
    
    
    
    pixel_spacing = float(root.get('MicronsPerPixel'))
    
    labels = []
    coords = []
    length = []
    area = []
    
    
    for r in regions:
        area += [float(r.get('AreaMicrons'))]
        length += [float(r.get('LengthMicrons'))]
        try:
            label = r[0][0].get('Value')
        except:
            label = r.get('Text')
        if 'benign' in label.lower():
            label = 1
        elif 'in situ' in label.lower():
            label = 2
        elif 'invasive' in label.lower():
            label = 3
        
        labels += [label]
        vertices = r[1]
        coord = []
        for v in vertices:
            x = int(v.get('X'))
            y = int(v.get('Y'))
            # print("(X,Y)",x,y)
            coord += [[x,y]]
    
        coords += [coord]

    return coords,labels,length,area,pixel_spacing

def get_masked_img(image_size,coordinates,labels):
    #red is 'benign', green is 'in situ' and blue is 'invasive'
    colors = [(0,0,0),(255,0,0),(0,255,0),(0,0,255)]
    
    img = np.zeros(image_size,dtype=np.uint8)
    
    for c,l in zip(coordinates,labels):
        img1 = fillImage(img,[np.int32(np.stack(c))],color=colors[l])
    return img1

def fillImage(image, coordinates,color=255):
    cv2.fillPoly(image, coordinates, color=color)
    return image


def save_labels_for_patch(patch_mask,patch,patch_np,file_name,r,c):
    label_wise_save_directory = args.save_path
    classes = ['Normal', 'Benign', 'InSitu', 'Invasive', 'Benign_InSitu', 'Benign_Invasive', 'InSitu_Invasive', 'Benign_InSitu_Invasive']
    reqd_path = {}
    for i,cl in enumerate(classes): 
        reqd_path[cl] = os.path.join(label_wise_save_directory,cl,file_name)
        os.makedirs(reqd_path[cl],exist_ok=True)
        if i<4:
            reqd_path[cl+"_impure"] = os.path.join(label_wise_save_directory,cl+"_impure",file_name)
            os.makedirs(reqd_path[cl+"_impure"],exist_ok=True)
    stats = np.unique(patch_mask,return_counts=True) 
    stats_R = np.unique(patch_mask[:,:,0],return_counts=True)
    stats_G = np.unique(patch_mask[:,:,1],return_counts=True)
    stats_B = np.unique(patch_mask[:,:,2],return_counts=True)
    if len(stats[0])==1 and stats[0] == 0 : 
        fp = reqd_path['Normal']+'/{}_X_{}_Y.png'.format(r,c) 
        patch.save(fp)
    else:
        if len(stats_R[0])==1 and  stats_R[0] == 255:
            fp = reqd_path['Benign']+'/{}_X_{}_Y.png'.format(r,c)
            patch.save(fp)
        elif len(stats_G[0])==1 and  stats_G[0] == 255:
            fp = reqd_path['InSitu']+'/{}_X_{}_Y.png'.format(r,c)
            patch.save(fp)
        elif len(stats_B[0])==1 and stats_B[0] == 255:
            fp = reqd_path["Invasive"]+'/{}_X_{}_Y.png'.format(r,c)
            patch.save(fp)
        else:
            if len(stats_R[0])>1 and len(stats_G[0])>1 and len(stats_B[0])>1 :
                fp = reqd_path['Benign_InSitu_Invasive']+'/{}_X_{}_Y.png'.format(r,c)
                patch.save(fp)
            elif len(stats_R[0])>1 and len(stats_G[0])>1 :
                fp = reqd_path['Benign_InSitu']+'/{}_X_{}_Y.png'.format(r,c)
                patch.save(fp)
            elif len(stats_R[0])>1 and len(stats_B[0])>1:
                fp = reqd_path['Benign_Invasive']+'/{}_X_{}_Y.png'.format(r,c)
                patch.save(fp)
            elif len(stats_G[0])>1 and len(stats_B[0])>1:
                fp = reqd_path['InSitu_Invasive']+'/{}_X_{}_Y.png'.format(r,c)
                patch.save(fp)
            elif len(stats_R[0])>1:
                purity = stats_R[1][1]/(256*256)
                if purity>=0.45:
                    fp = reqd_path['Benign']+'/{}_X_{}_Y.png'.format(r,c)
                    patch.save(fp)
                else:
                    fp = reqd_path['Benign_impure']+'/{}_X_{}_Y.png'.format(r,c)
                    patch.save(fp)

            elif len(stats_G[0])>1:
                purity = stats_G[1][1]/(256*256)
                if purity>=0.45:
                    fp = reqd_path['InSitu']+'/{}_X_{}_Y.png'.format(r,c)
                    patch.save(fp)
                else:
                    fp = reqd_path['InSitu_impure']+'/{}_X_{}_Y.png'.format(r,c)
                    patch.save(fp)
            elif len(stats_B[0])>1:
                purity = stats_B[1][1]/(256*256)
                if purity>=0.45:
                    fp = reqd_path['Invasive']+'/{}_X_{}_Y.png'.format(r,c)
                    patch.save(fp)
                else:
                    fp = reqd_path['Invasive_impure']+'/{}_X_{}_Y.png'.format(r,c)
                    patch.save(fp)

            else:
                print("total stats",stats)
                print("R stats",stats_R)
                print("G stats",stats_G)
                print("B stats",stats_B)

def save_patches_256(slide_file,xml_file):
    label_wise_save_directory = args.save_path
    whiteness_limit = 210
    blackness_limit = 5
    max_faulty_pixels = 0.8
    min_conn_comp = 10
    scan = openslide.OpenSlide(slide_file)
    orig_w,orig_h = scan.level_dimensions[0]
    file_name = slide_file.split('/')[-1].split('.')[0]
    patch_size = 256
    dims = scan.dimensions
    img_size = (dims[1],dims[0],3)
    
    coords,labels,length,area,pixel_spacing = readXML(xml_file)
    masked_img = get_masked_img(img_size,coords,labels)    
    os.makedirs("/ssd_scratch/cvit/ashishmenon/ICIAR2018_reconstructed_slides/",exist_ok=True)

    # slide_img_to_be_saved = np.ones((orig_w,orig_h,3),dtype=np.uint8)*240
    # masked_img_to_be_saved = np.zeros(masked_img.shape,dtype=np.uint8)    
    # for r in range(0,orig_w,patch_size):
    #     for c in range(0, orig_h,patch_size):
    #         if c+patch_size > orig_h and r+patch_size<= orig_w:
    #             p = orig_h-c
    #             pp = patch_size

    #         elif c+patch_size <= orig_h and r+patch_size > orig_w:
    #             p = patch_size
    #             pp = orig_w-r

    #         elif  c+patch_size > orig_h and r+patch_size > orig_w:
    #             p = orig_h-c
    #             pp = orig_w-r
                
    #         else:
    #             p = patch_size
    #             pp = patch_size    
    #         patch_to_be_saved = np.swapaxes(np.array(scan.read_region((r,c),0,(p,pp)),dtype=np.uint8)[...,0:3],1,0)
    #         patch_mask = masked_img[c:c+p,r:r+pp]
    #         is_white = np.all([patch_to_be_saved[:,:,i]>whiteness_limit for i in range(3)], axis=0)
    #         is_black = np.all([patch_to_be_saved[:,:,i]<blackness_limit for i in range(3)], axis=0)
    #         num_labels, labels_im = get_connected_components(patch_to_be_saved)
    #         if np.sum(is_white+is_black)>patch_size*patch_size*max_faulty_pixels and num_labels<min_conn_comp:
    #             continue
    #         else:
    #             masked_img_to_be_saved[c:c+p,r:r+pp] = patch_mask
    #             slide_img_to_be_saved[r:r+pp,c:c+p] = patch_to_be_saved
    #             save_labels_for_patch(patch_mask,patch_to_be_saved,file_name,r,c)
                
    upscale_factor = 1
    patch_size_upscaled = upscale_factor*patch_size
    W,H = orig_w-orig_w%patch_size_upscaled,orig_h-orig_h%patch_size_upscaled
    # slide_img_to_be_saved = np.zeros((W//upscale_factor,H//upscale_factor,3),dtype=np.uint8)
    # masked_img_to_be_saved = np.zeros((H//upscale_factor,W//upscale_factor,3),dtype=np.uint8)
    for r in range(0,W//upscale_factor,patch_size):
        for c in range(0, H//upscale_factor,patch_size):
            print(r,c)
            patch = scan.read_region(location=(upscale_factor*r,upscale_factor*c), level=0, \
                        size=(patch_size_upscaled,patch_size_upscaled)).convert('RGB')
            patch = patch.resize((patch_size,patch_size))
            patch_np = np.swapaxes(np.array(patch),1,0)
            
            patch_mask_upscaled = masked_img[upscale_factor*c:(upscale_factor*c)+patch_size_upscaled,upscale_factor*r:(upscale_factor*r)+patch_size_upscaled]
            try:
                patch_mask = cv2.resize(patch_mask_upscaled, (patch_size, patch_size), interpolation=cv2.INTER_CUBIC)
            except:
                print("Exception",patch_mask_upscaled.shape)
            is_white = np.all([patch_np[:,:,i]>whiteness_limit for i in range(3)], axis=0)
            is_black = np.all([patch_np[:,:,i]<blackness_limit for i in range(3)], axis=0)
            num_labels, labels_im = get_connected_components(patch_np)
            if np.sum(is_white+is_black)>patch_size*patch_size*max_faulty_pixels and num_labels<min_conn_comp:
                continue
            else:
                save_labels_for_patch(patch_mask,patch,patch_np,file_name,r,c)
                # slide_img_to_be_saved[r:r+patch_size,c:c+patch_size] = patch_np
                # masked_img_to_be_saved[c:c+patch_size,r:r+patch_size] = patch_mask              
                    
            
    print("Done completely")
    # Image.fromarray(slide_img_to_be_saved[::8,::8]).save("/ssd_scratch/cvit/ashishmenon/ICIAR2018_reconstructed_slides/"+"{}_resized_8.png".format(file_name))
    # Image.fromarray(masked_img_to_be_saved[::8,::8]).save("/ssd_scratch/cvit/ashishmenon/ICIAR2018_reconstructed_slides/"+"{}_mask_resized_8.png".format(file_name))
    # Image.fromarray(np.swapaxes(slide_img_to_be_saved,1,0)[::8,::8]).save("/ssd_scratch/cvit/ashishmenon/ICIAR2018_reconstructed_slides/"+"{}_resized_8_swapped.png".format(file_name))
    # Image.fromarray(np.swapaxes(masked_img_to_be_saved,1,0)[::8,::8]).save("/ssd_scratch/cvit/ashishmenon/ICIAR2018_reconstructed_slides/"+"{}_mask_resized_8_swapped.png".format(file_name))
    

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, default="./")
    parser.add_argument('--save_path', type=str, default='./')

    args = parser.parse_args()
    
    reqd_patients = ['A01','A02','A03','A04','A05','A06','A07','A08','A09','A10']
    # reqd_patients= [x.split('/')[-1].split('.')[0] for x in slides_list]
    
    slides_list = glob.glob(args.dataroot+'/*.svs')
    
    ann_list = glob.glob(args.dataroot+ '/*.xml')

    reqd_ann = [x for x in ann_list if x.split('/')[-1].split('.')[0] in reqd_patients]
    reqd_slides = [x for x in slides_list if x.split('/')[-1].split('.')[0] in reqd_patients]


    reqd_ann.sort()
    reqd_slides.sort()

    print(len(reqd_ann),len(reqd_slides))
    for i in zip(reqd_slides,reqd_ann):
        print(i[0].split('/')[-1],i[1].split('/')[-1])
    pool = Pool(multiprocessing.cpu_count())
    pool.starmap(save_patches_256, zip(reqd_slides,reqd_ann))



