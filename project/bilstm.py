# %% [markdown]
# Import Statements


# %%
import torch
import torch.nn as nn
from datasets import load_dataset

from typing import List, Tuple
from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict
MyDatasetType = Dataset | DatasetDict | IterableDataset | IterableDatasetDict
import time

from utils import is_projective, evaluate, parse_step
from arceagerparser import (
    ArcEager,
    Oracle,
    generate_gold,
    NOMOVE,
    RIGHT_ARC,
    LEFT_ARC,
    SHIFT,
    REDUCE,
    is_right_possible,
    is_left_possible,
    is_shift_possible,
    is_reduce_possible,
)
# %% [markdown]
# Network Parameters

# %%
class NNParameters():
  def __init__(self) -> None:
      self.BATCH_SIZE = 2
      self.EMBEDDING_SIZE = 200
      self.LSTM_SIZE = 200
      self.LSTM_LAYERS = 2
      self.MLP_INPUT_SIZE = self.LSTM_LAYERS * self.LSTM_SIZE
      self.OUT_CLASSES = 4
      
      self.DROP_OUT = 0.2
      self.LR = 0.001
      self.EPOCHS = 30

nn_params = NNParameters()

class NNData():
  def __init__(self, tokens, confs, moves, heads, dictionary) -> None:
      self.tokens = tokens
      self.confs = confs
      self.moves = moves
      self.heads = heads
      self.dictionary = dictionary


      

# %% [markdown]
# Function Definitions


# %%
def create_dictionary(dataset: MyDatasetType , threshold: int =3) -> dict[str, int]:
    """
    Extract from corpus vocabulary V of unique words that appear at least threshold times.
    input:
        dataset: list of sentences, each sentence is a list of words
        treashold: minimum number of times a word must appear in the corpus to be included in the vocabulary
        
    output:
        map: dictionary of word/index pairs. This is our embedding list
    """
    dic = {}  # dictionary of word counts
    for sample in dataset:
        for word in sample["tokens"]:
            if word in dic:
                dic[word] += 1
            else:
                dic[word] = 1

    map = {}  # dictionary of word/index pairs. This is our embedding list
    map["<pad>"] = 0
    map["<ROOT>"] = 1
    map["<unk>"] = 2  # used for words that do not appear in our list

    next_indx = 3
    for word in dic.keys():
        if dic[word] >= threshold:
            map[word] = next_indx
            next_indx += 1

    return map


def process_sample(db, emb_dictionary, get_gold_path=False):
    """
    Process a sample from the dataset
    1. Add ["<ROOT>"] to the beginning of the sentence and [-1] to the beginning of the head
    2. Encode the sentence and the gold path
    
    
    :param         tokens: tokens of a sentence
    :param emb_dictionary: dictionary of word/index pairs
    :param  get_gold_path: if True, we also return the gold path and gold moves
    :return: enc_sentence: encoded tokens of the sentence
                gold_path: gold path of the sentence
               gold_moves: gold moves of the sentence
                     gold: gold heads of the sentence
    """
    sentence = ["<ROOT>"] + db["tokens"]
    head = [-1] + list(map(int, db["head"]))  # [int(i) for i in tokens["head"]]

    # embedding ids of sentence words
    enc_sentence = [
        emb_dictionary[word] if word in emb_dictionary else emb_dictionary["<unk>"]
        for word in sentence
    ]


    if get_gold_path:
        gold_moves, gold_path, _= generate_gold(sentence, head) # transform matrix from nx3 to 3xn
    else:
        gold_path, gold_moves = [], []
    
    return enc_sentence, gold_path, gold_moves, head


def process_batch(batch:List[List], emb_dictionary:dict[str,int], get_gold_path:bool=False) -> NNData:
  sentences = []
  confs = []
  moves = []
  heads = []
  for sample in batch:
    s, c, m, h= process_sample(sample, emb_dictionary, get_gold_path=get_gold_path)
    sentences.append(s)
    confs.append(c)
    moves.append(m)
    heads.append(h)
    
  processed_data = NNData(sentences, confs, moves, heads, emb_dictionary)
  
  return processed_data


# %% [markdown]
# Network definition

# %%


# %% [markdown]
# Main Function


# %%

if __name__ == "__main__":
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print(device)
  torch.manual_seed(99)
  
  ## Download data
  train_dataset = load_dataset("universal_dependencies", "en_lines", split="train[:200]")
  test_dataset = load_dataset("universal_dependencies", "en_lines", split="test[:100]")
  validation_dataset = load_dataset("universal_dependencies", "en_lines", split="validation[:100]")
  print(
      f"train_dataset: {len(train_dataset)}, validation_dataset: {len(validation_dataset)}, test_dataset: {len(test_dataset)}" # type:ignore
  )  

  train_dataset = train_dataset.filter(lambda x: is_projective([-1] + list(map(int, x["head"]))))
  validation_dataset = validation_dataset.filter(lambda x: is_projective([-1] + list(map(int, x["head"]))))
  test_dataset = test_dataset.filter(lambda x: is_projective([-1] + list(map(int, x["head"]))))
  print(
      f"PROJECTIVE -> train_dataset: {len(train_dataset)}, validation_dataset: {len(validation_dataset)}, test_dataset: {len(test_dataset)}" # type:ignore
  )  
  
  ## Start processing data
  dictionary= create_dictionary(train_dataset)

  ## Dataloader
  train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=nn_params.BATCH_SIZE, 
    shuffle=True,
    collate_fn=lambda x: process_batch(x, dictionary, get_gold_path=True)
  )
  validation_dataloader= torch.utils.data.DataLoader(
    validation_dataset,
    batch_size=nn_params.BATCH_SIZE,
    shuffle=True,
    collate_fn=lambda x: process_batch(x, dictionary, get_gold_path=True)
  )
  test_dataloader= torch.utils.data.DataLoader(
    test_dataset,
    batch_size=nn_params.BATCH_SIZE,
    shuffle=True,
    collate_fn=lambda x: process_batch(x, dictionary, get_gold_path=False)
  )
    
  for b in train_dataloader: 
    print("##############################################")
    print(b.tokens)
    print(b.confs)
    print(b.moves)
    print(b.heads)
    
  


# %%
