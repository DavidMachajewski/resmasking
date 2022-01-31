from rmn import RMN, models
from pathlib import Path
from configs import Arguments
from lib.loader import get_loaders
from lib.pipeline import JAFFETrainer


def train_network(args, config):
    traindl, testdl, valdl = get_loaders(args)

    model = models.__dict__[config["arch"]]

    trainer = JAFFETrainer(model=model, train_set=traindl, val_set=valdl, test_set=testdl, configs=config)
    trainer.train()


if __name__ == '__main__':
    arguments = Arguments()
    args = arguments.parser
    cnf = arguments.cnf_fer

    train_network(args, cnf)


# /home/shared/AffectiveAI/data/fer2013/train_ids_0.csv
# CUDA_VISIBLE_DEVICES=0 python main.py --DATASET "FER" --FER2013PTR "D:\datasets\fer\fer\train_ids_0.csv" --FER2013PTE "D:\datasets\fer\fer\test_ids_0.csv" --FER2013PVA "D:\datasets\fer\fer\valid_ids_0.csv"



# CUDA_VISIBLE_DEVICES=3 python main.py --DATASET "FER" --FER2013PTR "/home/shared/AffectiveAI/data/fer2013/train_ids_0.csv" --FER2013PTE "/home/shared/AffectiveAI/data/fer2013/test_ids_0.csv" --FER2013PVA"


#
# Set-ExecutionPolicy Unrestricted -Scope Process
# activate
# python main.py --DATASET "FER" --FER2013PTR "D:\datasets\fer\fer\train_ids_0.csv" --FER2013PTE "
# D:\datasets\fer\fer\test_ids_0.csv" --FER2013PVA "D:\datasets\fer\fer\valid_ids_0.csv"
#
#
# pip3 install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio===0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html