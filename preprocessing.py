from lib.loader import get_transforms,FER2013
from configs import Arguments
from PIL import Image
from pathlib import Path
import numpy as np


def create_name(idx, label) -> str:
    return f"{idx}_{label}.tiff"


if __name__ == '__main__':
    arguments = Arguments()
    args = arguments.parser
    cnf = arguments.cnf_fer

    transformer = get_transforms([224, 224])
    testset = FER2013(args=args, mode="train", transform=transformer)

    for idx, sample in enumerate(testset):
        image = sample['image'].permute((1,2,0)).numpy()
        label = sample['label']

        filename = create_name(idx, label)
        path = Path(args.PREP_SAVE_TO) / filename

        im = Image.fromarray(np.uint8(image))
        im.save(path)
        if idx == 149:
            break

# python preprocessing.py --DATASET "FER" --FER2013PTR "D:\datasets\fer\fer\train_ids_0.csv" --FER2013PTE
#  "D:\datasets\fer\fer\test_ids_0.csv" --FER2013PVA "D:\datasets\fer\fer\valid_ids_0.csv" --PREP_SAVE_TO "E:\resmaskingfer2013\imgs
# "