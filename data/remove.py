import os
from pathlib import Path
import tqdm


def remove_non_npy_jpeg(data_path):
    dataset_name = data_path.split("/")[-1]
    root = Path(data_path)
    num_files = len(list(root.glob("**/*_cropped.jpg")))

    for idx, img_path in tqdm.tqdm(enumerate(root.glob("**/*_cropped.jpg")), total=num_files):
        split_path = str(img_path).split("/")
        true_name = split_path[-2]
        parent_dir = "/".join(split_path[:-1])

        for file_path in Path(parent_dir).glob("*"):
            if true_name + '_cropped_uv_posmap.npy' not in str(file_path) and \
               true_name + '_cropped.jpg' not in str(file_path):
               file_path.unlink()


if __name__=="__main__":
    remove_non_npy_jpeg("data/images/300W_LP-crop")
    remove_non_npy_jpeg("data/images/AFLW2000-crop")
    # remove_non_npy_jpeg("images/test")

