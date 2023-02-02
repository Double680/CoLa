## Instruction for Dataset Pre-processing

1. Download corresponding pre-trained ResNets for the two datasets according to the description in the formal paper.
2. Extract the concatenated FC weights and save them as "WNV-2K/FCWeights.pth" or "WNV-13K/FCWeights.pth".
3. Generate GloVe embeddings (already exists, which are generated as the same to "glove.py" in DGP's official code).
4. Copy "n-n.py" into the data folder and then execute it. (Caution: Create "valid.py" file first with any triplets in the same format to "train.py", which is meaningless and never used in our ZSL study. It is just for successfully executing file "n-n.py")