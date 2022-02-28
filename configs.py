import argparse


class Arguments:
    def __init__(self):
        self.parser: argparse.ArgumentParser() = self.__setup__()
        self.cnf_fer = {
            "data_path": "content/fer2013/dataset/",
            "image_size": 224,
            "in_channels": 3,
            "num_classes": 7,
            "arch": "resmasking_dropout1",  # alexnet
            "lr": 0.001,
            "weighted_loss": 0,
            "momentum": 0.9,
            "weight_decay": 0.001,
            "distributed": 0,
            "batch_size": 48,
            "num_workers": 8,
            "device": "cuda:0",  # "cuda:0", # "cpu"
            "max_epoch_num": 50,
            "max_plateau_count": 8,
            "plateau_patience": 3,
            "steplr": 50,
            "log_dir": "saved/logs/",
            "checkpoint_dir": "saved/checkpoints/",
            "model_name": "test",
            "cwd": "content/"
        }

    def __setup__(self) -> argparse.ArgumentParser():
        parser = argparse.ArgumentParser(description='Run Load', epilog="Thanks for using PF dataloads.")

        # # # training args # # #
        parser.add_argument('--DATASET', default="FER", type=str)

        # # # dataset paths # # #
        parser.add_argument('--FER2013PTR', type=str, help="PATH TO THE TRAINING SPLIT OF FER2013")
        parser.add_argument('--FER2013PTE', type=str, help="PATH TO THE TESTING SPLIT OF FER2013")
        parser.add_argument('--FER2013PVA', type=str, help="PATH TO THE VALIDATION SPLIT OF FER2013")

        # # # preprocessing # # #
        parser.add_argument('--PREP_SAVE_TO', type=str)

        return parser.parse_args()