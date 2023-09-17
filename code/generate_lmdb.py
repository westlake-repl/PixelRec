import os
from PIL import Image
import numpy as np
import lmdb
import pandas as pd
import pickle
import tqdm
import torch
import torchvision.transforms as transforms

torch.manual_seed(123456)
'''
bulid lmdb database from images.
'''

interaction_path = '../dataset/Pixel200K.csv'
image_path = '../images/'
generate_lmdb_name = '../dataset/image.lmdb'

class LMDB_Image:
    def __init__(self, image, id):
        self.channels = image.shape[2]
        self.size = image.shape[:2]
        self.image = image.tobytes()
        self.id = id

    def get_image(self):
        image = np.frombuffer(self.image, dtype=np.uint8)
        return image.reshape(*self.size, self.channels)


if __name__ == '__main__':
    print('build lmdb database')
    interaction = pd.read_csv(interaction_path, usecols=[0])
    items = interaction['item_id'].unique()
    image_num = len(items)
    print("all images %s" % image_num)

    lmdb_path = generate_lmdb_name
    isdir = os.path.isdir(lmdb_path)
    print("Generate LMDB to %s" % lmdb_path)
    lmdb_env = lmdb.open(lmdb_path, subdir=isdir, map_size=image_num * np.zeros((3, 224, 224)).nbytes*10,
                         readonly=False, meminit=False, map_async=True)
    txn = lmdb_env.begin(write=True)
    write_frequency = 5000


    bad_file = {}
    t = transforms.Resize((224,224))     
    lmdb_keys = []
    for index, row in enumerate(tqdm.tqdm(items)):
        item_id = str(row)
        item_name = str(row)+ '.jpg'
        lmdb_keys.append(item_id)
        try:
            img = Image.open(os.path.join(image_path, item_name)).convert('RGB')
            img = t(img)  #resize the image to (3,224,224) before stored into database,you can remove this if you don't need it.
            img = np.array(img)
            temp = LMDB_Image(img, item_id)
            txn.put(u'{}'.format(item_id).encode('ascii'), pickle.dumps(temp))
            if index % write_frequency == 0 and index != 0:
                txn.commit()
                txn = lmdb_env.begin(write=True)
        except Exception as e:
            bad_file[index] = item_id

    txn.commit()
    keys = [u'{}'.format(k).encode('ascii') for k in lmdb_keys]
    with lmdb_env.begin(write=True) as txn:
        txn.put(b'__keys__', pickle.dumps(keys))
        txn.put(b'__len__', pickle.dumps(len(keys)))
    print(len(keys))
    print("Flushing database ...")
    lmdb_env.sync()
    lmdb_env.close()
    print(f'bad_file: {len(bad_file)}')
