import zlib
import pickle
from pathlib import Path
import tqdm
import os
import numpy as np
from zipfile import ZipFile, ZIP_DEFLATED

def compress_data(data, filepath="data/compressed-data/300WLP.gz"):
    with open(filepath, 'ab') as f:
        f.write(zlib.compress(pickle.dumps(data, pickle.HIGHEST_PROTOCOL), 9))

def decompress_data(filepath="data/compressed-data/300WLP.gz"):
    with open(filepath, 'rb') as fp:

        data = zlib.decompress(fp.read())
        successDict = pickle.loads(data)
    return successDict

root = Path("data/images/300W_LP-crop")
os.makedirs("data/compressed-data", exist_ok=True)
num_files = len(list(root.glob("**/*_cropped.jpg")))
data = {}

with ZipFile('sample.zip', 'w', ZIP_DEFLATED) as zip_obj:
    for idx, img_path in tqdm.tqdm(enumerate(root.glob("**/*_cropped.jpg")), total=num_files):
        split_path = str(img_path).split("/")
        dataset = split_path[3]
        true_name = split_path[-2]

        parent_dir = "/".join(split_path[:-1])
        uv_path = os.path.join(parent_dir, true_name + '_cropped_uv_posmap.npy')
        zip_obj.write(uv_path)
        # data[true_name] = np.load(uv_path, allow_pickle=True)

        # compress_data(data)
        # import ipdb; ipdb.set_trace(context=10)
        break
    zip_obj.close()