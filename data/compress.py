import os
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import tqdm

def compress_PRnet_data(data_path, save_path):
    dataset_name = data_path.split("/")[-1]
    root = Path(data_path)
    num_files = len(list(root.glob("**/*_cropped.jpg")))

    with ZipFile(os.path.join(save_path, dataset_name+".zip"), 'w', ZIP_DEFLATED) as zip_obj:
        for idx, img_path in tqdm.tqdm(enumerate(root.glob("**/*_cropped.jpg")), total=num_files):
            split_path = str(img_path).split("/")
            dataset = split_path[3]
            true_name = split_path[-2]
            parent_dir = "/".join(split_path[:-1])

            uv_path = os.path.join(parent_dir, true_name + '_cropped_uv_posmap.npy')

            zip_obj.write(str(img_path))
            zip_obj.write(uv_path)

            img_path.unlink()
            Path(uv_path).unlink()
        zip_obj.close()

if __name__=="__main__":
    os.makedirs("data/compressed-data", exist_ok=True)
    print("Process 300WLP:")
    compress_PRnet_data("data/images/300W_LP-crop", "data/compressed-data")
    print("Process AFLW2000:")
    compress_PRnet_data("data/images/AFLW2000-crop", "data/compressed-data")