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

from utils import is_projective, evaluate, parse_moves
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
      self.BATCH_SIZE = 128
      self.EMBEDDING_SIZE = 200
      self.LSTM_SIZE = 200
      self.LSTM_LAYERS = 2
      self.MLP_OUT_SIZE = self.LSTM_LAYERS * self.LSTM_SIZE
      self.OUT_CLASSES = 4
      
      self.DROP_OUT = 0.2
      self.LR = 0.001
      self.EPOCHS = 30

nnp = NNParameters()

class NNData():
  def __init__(self, tokens, confs, moves, heads) -> None:
      self.enc_tokens = tokens
      self.confs = confs
      self.moves = moves
      self.heads = heads
      #self.dictionary = dictionary

def extract_att(data:List[NNData], attribute:str):
  return [getattr(d, attribute) for d in data]
      

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


def process_sample(sample, emb_dictionary, get_gold_path=False):
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
    sentence = ["<ROOT>"] + sample["tokens"]
    head = [(-1)] + list(map(int, sample["head"]))  # [int(i) for i in tokens["head"]]

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


def process_batch(batch:List[List], emb_dictionary:dict[str,int], get_gold_path:bool=False) -> List[NNData]:
  pack:List[NNData]=[]

  for sample in batch:
    s, c, m, h= process_sample(sample, emb_dictionary, get_gold_path=get_gold_path)
    pack.append(NNData(s, c, m, h))

  return pack 


# %% [markdown]
# Network definition

# %%
class BiLSTMNet(nn.Module):
  def __init__(self,device, dictionary,  *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self.device = device
    self.embeddings = nn.Embedding(
      len(dictionary),
      nnp.EMBEDDING_SIZE,
      padding_idx=dictionary["<pad>"]
    )

    self.lstm = nn.LSTM(
      nnp.EMBEDDING_SIZE,
      nnp.LSTM_SIZE,
      num_layers=nnp.LSTM_LAYERS,
      bidirectional=True,
      dropout=nnp.DROP_OUT,
    )
    
    self.w1 = nn.Linear(2 * nnp.LSTM_LAYERS * nnp.LSTM_SIZE, nnp.MLP_OUT_SIZE, bias=True)
    self.activation = nn.Tanh()
    self.w2 = nn.Linear(nnp.MLP_OUT_SIZE, nnp.OUT_CLASSES, bias=True)
    self.softmax = nn.Softmax(dim=-1)
    self.dropout = nn.Dropout(nnp.DROP_OUT)
  
  def get_mlp_input(self, configurations, h):
      mlp_input = []
      zero_tensor = torch.zeros(
          2 * nnp.LSTM_SIZE, requires_grad=False, device=self.device
      )
      for i in range(len(configurations)):
          for j in configurations[i]:  # for each configuration of a sentence
              mlp_input.append(
                  torch.cat(
                      [
                          zero_tensor if j[0] == -1 else h[j[0]][i],
                          zero_tensor if j[1] == -1 else h[j[1]][i],
                      ]
                  )
              )
      mlp_input = torch.stack(mlp_input).to(self.device)
      return mlp_input

  def mlp_pass(self, x):
      return self.softmax(
          self.w2(self.dropout(self.activation(self.w1(self.dropout(x)))))
      )

  def lstm_pass(self, x):
    x = torch.nn.utils.rnn.pack_sequence(x, enforce_sorted=False)
    h, _ = self.lstm(x)
    h, _ = torch.nn.utils.rnn.pad_packed_sequence(h)
    return h
  
  def forward(self, batch:List[NNData]):
    tokens = extract_att(batch, "enc_tokens")
    x = [self.dropout(self.embeddings(torch.tensor(t).to(self.device))) for t in tokens]

    h = self.lstm_pass(x)
    
    configurations:List[List[Tuple[int,int]]] = extract_att(batch, "confs")
    mlp_input = self.get_mlp_input(configurations, h)
    out = self.mlp_pass(mlp_input)
    return out

  def infere(self, batch):
    start_time=time.time()
    tokens=extract_att(batch, "enc_tokens")
    parsers: List[ArcEager] = [ArcEager(t) for t in tokens]

    x = [self.embeddings(torch.tensor(t).to(self.device)) for t in tokens]
    h = self.lstm_pass(x)
    print("time lstm", time.time()-start_time)

    start_time=time.time()
    is_final = [False] 
    while not all(is_final):
      # get the current configuration and score next moves
      configurations = [[p.get_configuration_now()] for p in parsers]
      mlp_input = self.get_mlp_input(configurations, h)
      mlp_out = self.mlp_pass(mlp_input)
      # take the next parsing step
      list_moves= parse_moves(parsers, mlp_out)
      for i,m in enumerate(list_moves):
          parsers[i].do_move(m)
      is_final=[t.is_tree_final() for t in parsers]
          
    print("time parse", time.time()-start_time)

    # return the predicted dependency tree
    return [parser.list_arcs for parser in parsers]
 



    
    

# %% [markdown]
# NN related functions

# %%
def train(model: BiLSTMNet, dataloader, criterion, optimizer):
  model.train()
  total_loss = 0
  count = 1
  for batch in dataloader:
    print(f"TRAIN: batch {count}/{len(dataloader):.0f}")
    optimizer.zero_grad()
    
    out = model(batch)

    moves= extract_att(batch, "moves")
    labels = torch.tensor(sum(moves, [])).to(
        device
    )  # sum(moves, []) flatten the array
    
    loss = criterion(out, labels)
    total_loss += loss.item()
    loss.backward()
    optimizer.step()
    count += 1

  return total_loss / count


def test(model: BiLSTMNet, dataloader: torch.utils.data.dataloader):  # type:ignore
  model.eval()

  gold = []
  preds = []
  count=0

  for batch in dataloader:
    print(f"test: batch {count}/{len(dataloader):.0f}")

    with torch.no_grad():
        pred = model.infere(batch)
        gold += extract_att(batch, "heads")
        preds += pred

  return evaluate(gold, preds)

# %% [markdown]
# Main Function


# %%

if __name__ == "__main__":
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print(device)
  torch.manual_seed(99)
  
  ## Download data
  train_dataset = load_dataset("universal_dependencies", "en_lines", split="train")
  test_dataset = load_dataset("universal_dependencies", "en_lines", split="test")
  validation_dataset = load_dataset("universal_dependencies", "en_lines", split="validation")
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
    batch_size=nnp.BATCH_SIZE, 
    shuffle=True,
    collate_fn=lambda x: process_batch(x, dictionary, get_gold_path=True)
  )
  validation_dataloader= torch.utils.data.DataLoader(
    validation_dataset,
    batch_size=nnp.BATCH_SIZE,
    shuffle=True,
    collate_fn=lambda x: process_batch(x, dictionary, get_gold_path=True)
  )
  test_dataloader= torch.utils.data.DataLoader(
    test_dataset,
    batch_size=nnp.BATCH_SIZE,
    shuffle=True,
    collate_fn=lambda x: process_batch(x, dictionary, get_gold_path=False)
  )
    
  # for b in train_dataloader: 
  #   print("##############################################")
  #   print(b.tokens)
  #   print(b.confs)
  #   print(b.moves)
  #   print(b.heads)
  
  
  model = BiLSTMNet(device, dictionary).to(device)
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=nnp.LR)
  
  for epoch in range(nnp.EPOCHS):
    print("Starting Epoch", epoch)
    # torch.load(f"bilstm_e{epoch+1}.pt")
    avg_train_loss = train(model, train_dataloader, criterion, optimizer)
    val_uas = test(model, validation_dataloader)

    log = f"Epoch: {epoch:3d} | avg_train_loss: {avg_train_loss:5.3f} | dev_uas: {val_uas:5.3f} |"
    print(log)

    # save the model on pytorch format
    torch.save(model.state_dict(), f"bilstm_e{epoch+1}.pt")

  test_uas = test(model, test_dataloader)
  log = "test_uas: {:5.3f}".format(test_uas)
  print(log)
  train(model, train_dataloader, criterion, optimizer)

  



# %%
