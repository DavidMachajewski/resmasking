# **ResidualMaskingNetwork**

---

This repo provides code for training the Residual Masking Network [1] on the FER2013 dataset [2].

---

### Training
To run the training on the FER2013 dataset use the command

> ````CUDA_VISIBLE_DEVICES=2 python main.py --DATASET "FER" --FER2013PTR "/home/shared/AffectiveAI/data/fer2013/train_ids_0.csv" --FER2013PTE "/home/shared/AffectiveAI/data/fer2013/test_ids_0.csv" --FER2013PVA "/home/shared/AffectiveAI/data/fer2013/valid_ids_0.csv"````

### Source
> [1] L. Pham, H. Vu, T. A. Tran, "Facial Expression Recognition Using Residual Masking Network", IEEE 25th International Conference on Pattern Recognition, 2020, 4513-4519. Milan -Italia. <br>
> [2] https://www.kaggle.com/datasets/msambare/fer2013