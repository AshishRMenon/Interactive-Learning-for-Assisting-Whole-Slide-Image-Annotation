import random
import glob
import os 
import shutil
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Args for Running Fb Retrieval from a search DB and a query DB")
    parser.add_argument("--train_path", type=str, required=True, help="Root location of images")
    args = parser.parse_args()



    classes = os.listdir(args.train_path)
    for i in classes:
        images_set = glob.glob(args.train_path+i+'/*.tif')
        query_set = random.sample(images_set,20)
        ann_set = random.sample(list(set(images_set)-set(query_set)),10)
        rem_set = list(set(images_set)-set(query_set)-set(ann_set))
        for a in ann_set:
            target = a.replace('/NCT-CRC-HE-100K',"")
            target = target.replace('/train/','/annotated/')
            target = ('/').join(target.split('/')[:-1])+"/"
            shutil.move(a,target)
        
        for q in query_set:
            target = q.replace('/NCT-CRC-HE-100K',"")
            target = target.replace('/train/','/query/')
            target = ('/').join(target.split('/')[:-1])+"/"
            shutil.move(q,target)
    