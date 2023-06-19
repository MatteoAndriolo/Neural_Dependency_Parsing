# %% [markdown]
# Import Statements


# %%
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from transformers.tokenization_utils_base import BatchEncoding

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
      self.BATCH_SIZE = 10
      self.BERT_SIZE = 768
      self.EMBEDDING_SIZE = self.BERT_SIZE
      self.DIM_CONFIG = 2
      self.MLP1_IN_SIZE = self.DIM_CONFIG * self.EMBEDDING_SIZE
      self.MLP2_IN_SIZE = 300
      self.OUT_CLASSES = 4
      
      self.DROP_OUT = 0.2
      self.LR = 0.001
      self.EPOCHS = 30

nnp = NNParameters()

class NNData():
  def __init__(self,sentence, confs, moves, heads, subw2word_idx) -> None:
      self.sentence = sentence
      self.confs = confs
      self.moves = moves
      self.heads = heads
      self.subw2word_idx = subw2word_idx

def extract_att(data:List[NNData], attribute:str):
  return [getattr(d, attribute) for d in data]
      

# %% [markdown]
# Function Definitions


# %%
def process_sample(sample, sentence, iid, get_gold_path=False):
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
    def calcualate_subw2word_idx(subw_idx):
      tokens=tokenizer.convert_ids_to_tokens(subw_idx)
      i=0
      o=[]
      while i<len(tokens):
        t=[]
        t.append(i)
        i+=1
        while i<len(tokens) and tokens[i].startswith("##"):
          t.append(i)
          i+=1
        o.append(t)
      return o

    head = [(-1)] + list(map(int, sample["head"]))  # [int(i) for i in tokens["head"]]

    # embedding ids of sentence words
    subw2word_idx=calcualate_subw2word_idx(iid)

    if get_gold_path:
        gold_moves, gold_path, _= generate_gold(sentence, head) # transform matrix from nx3 to 3xn
    else:
        gold_path, gold_moves = [], []

    return head, gold_path, gold_moves, subw2word_idx


def process_batch(batch:List[List], tokenizer, get_gold_path:bool=False) :
  pack:List[NNData]=[]
  sentences=[["<ROOT>"]+ bd["tokens"] for bd in batch]

  ## Tokenizer -> get token_ids, attention_mask and token_type_ids
  output_tokenizer: BatchEncoding= tokenizer(
      ["<ROOT> "+ bd["text"] for bd in batch],
      padding=True, 
      return_tensors="pt",
      add_special_tokens=False 
  )

  token_ids:List[List[int]]=output_tokenizer["input_ids"]
  attention_mask=output_tokenizer["attention_mask"]
  token_types_ids=output_tokenizer["token_type_ids"] 
  ###########
  ## What is left? heads, gold_path, gold_moves, subw2word_idx
  ## What i need? sample -> heads, original sentence -> golds, input_ids -> subw2word_idx
  for sample, sentence, iid in zip(batch,sentences,token_ids):
    head, configuration, moves, s2w= process_sample(sample, sentence,iid, get_gold_path=get_gold_path)
    pack.append(NNData(sentence, configuration, moves, head, s2w))

  return output_tokenizer, pack



# %% [markdown]
# Network definition

# %%

class BERTNet(nn.Module):
  def __init__(self,device, tokenizer,  *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self.device = device
    self.embeddings = nn.Embedding(
      len(tokenizer),
      nnp.EMBEDDING_SIZE,
      padding_idx=0
    )

    
    self.bert = AutoModel.from_pretrained('bert-base-uncased')
    self.bert.resize_token_embeddings(len(tokenizer))
    
    # Freeze bert layers
    for param in self.bert.parameters():
      param.requires_grad = False
    
    self.w1 = nn.Linear(nnp.MLP1_IN_SIZE, nnp.MLP2_IN_SIZE, bias=True)
    self.activation = nn.Tanh()
    self.w2 = nn.Linear(nnp.MLP2_IN_SIZE, nnp.OUT_CLASSES, bias=True)
    self.softmax = nn.Softmax(dim=-1)
    self.dropout = nn.Dropout(nnp.DROP_OUT)
  
  def get_embedding(self, h, idx):
      return torch.mean(h[idx], dim=0)

  def get_mlp_input(self, configurations, subw2idx, h):
      mlp_input = []
      zero_tensor = torch.zeros(
          nnp.BERT_SIZE, requires_grad=False, device=self.device
      )
      for i in range(len(configurations)):
          for j in configurations[i]:  # for each configuration of a sentence
              mlp_input.append(
                  torch.cat(
                      [
                          zero_tensor if j[0] == -1 else self.get_embedding(h[i], subw2idx[i][j[0]]),
                          zero_tensor if j[1] == -1 else self.get_embedding(h[i], subw2idx[i][j[1]]),
                      ]
                  )
              )
      mlp_input = torch.stack(mlp_input).to(self.device)
      return mlp_input

  def mlp_pass(self, x):
      return self.softmax(
          self.w2(self.dropout(self.activation(self.w1(self.dropout(x)))))
      )

  
  def forward(self, batch:Tuple[BatchEncoding,List[NNData]]):
    output_tokenizer = batch[0].to(self.device)
    input_ids= output_tokenizer["input_ids"]
    attention_mask= output_tokenizer["attention_mask"] 
    input_ids=input_ids.to(self.device)
    attention_mask=attention_mask.to(self.device)

    h = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state.to(self.device)
    
    nndata=batch[1]
    configurations = extract_att(nndata, "confs")
    subw2idx = extract_att(nndata, "subw2word_idx")
    mlp_input = self.get_mlp_input(configurations,subw2idx, h)

    out = self.mlp_pass(mlp_input)
    return out

  def infere(self, batch):
    output_tokenizer, nndata= batch
    output_tokenizer=output_tokenizer.to(self.device)
    input_ids= output_tokenizer["input_ids"]
    attention_mask= output_tokenizer["attention_mask"] 
    input_ids=input_ids.to(self.device)
    attention_mask=attention_mask.to(self.device)

    h = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state.to(self.device)

    tokens = extract_att(nndata, "sentence")
    parsers: List[ArcEager] = [ArcEager(t) for t in tokens]


    subw2idx = extract_att(nndata, "subw2word_idx")
    is_final = [False] 
    while not all(is_final):
      # get the current configuration and score next moves
      configurations = [[p.get_configuration_now()] for p in parsers]
      mlp_input = self.get_mlp_input(configurations, subw2idx, h)
      mlp_out = self.mlp_pass(mlp_input)
      # take the next parsing step
      list_moves= parse_moves(parsers, mlp_out)
      for i,m in enumerate(list_moves):
          parsers[i].do_move(m)
      is_final=[t.is_tree_final() for t in parsers]
          

    # return the predicted dependency tree
    return [parser.list_arcs for parser in parsers]
 



    
    

# %% [markdown]
# NN related functions

# %%
def train(model: BERTNet, dataloader, criterion, optimizer):
  model.train()
  total_loss = 0
  count = 1
  for batch in dataloader:
    print(f"TRAIN: batch {count}/{len(dataloader):.0f}")
    optimizer.zero_grad()
    
    out = model(batch)
    _, nndata=batch

    moves= extract_att(nndata, "moves")
    labels = torch.tensor(sum(moves, [])).to(
        device
    )  # sum(moves, []) flatten the array
    
    loss = criterion(out, labels)
    total_loss += loss.item()
    loss.backward()
    optimizer.step()
    count += 1

  return total_loss / count


def test(model: BERTNet, dataloader: torch.utils.data.dataloader):  # type:ignore
  model.eval()

  gold = []
  preds = []
  count=0

  for batch in dataloader:
    _, nndata=batch
    print(f"test: batch {count}/{len(dataloader):.0f}")

    with torch.no_grad():
        pred = model.infere(batch)
        gold += extract_att(nndata, "heads")
        preds += pred

  return evaluate(gold, preds)

# %% [markdown]
# Main Function


# %%

if __name__ == "__main__":
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print(device)
  torch.manual_seed(99)

  tokenizer=AutoTokenizer.from_pretrained('bert-base-uncased')
  tokenizer.add_tokens(["<ROOT>", "<EMPTY>"], special_tokens=True)
  
  ## Download data
  train_dataset = load_dataset("universal_dependencies", "en_lines", split="train[:100]")
  test_dataset = load_dataset("universal_dependencies", "en_lines", split="test[:50]")
  validation_dataset = load_dataset("universal_dependencies", "en_lines", split="validation[:50]")
  print(
      f"train_dataset: {len(train_dataset)}, validation_dataset: {len(validation_dataset)}, test_dataset: {len(test_dataset)}" # type:ignore
  )  

  train_dataset = train_dataset.filter(lambda x: is_projective([-1] + list(map(int, x["head"]))))
  validation_dataset = validation_dataset.filter(lambda x: is_projective([-1] + list(map(int, x["head"]))))
  test_dataset = test_dataset.filter(lambda x: is_projective([-1] + list(map(int, x["head"]))))
  print(
      f"PROJECTIVE -> train_dataset: {len(train_dataset)}, validation_dataset: {len(validation_dataset)}, test_dataset: {len(test_dataset)}" # type:ignore
  )  
  
  ## Dataloader
  train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=nnp.BATCH_SIZE, 
    shuffle=True,
    collate_fn=lambda x: process_batch(x, tokenizer, get_gold_path=True)
  )
  validation_dataloader= torch.utils.data.DataLoader(
    validation_dataset,
    batch_size=nnp.BATCH_SIZE,
    shuffle=True,
    collate_fn=lambda x: process_batch(x,tokenizer, get_gold_path=True)
  )
  test_dataloader= torch.utils.data.DataLoader(
    test_dataset,
    batch_size=nnp.BATCH_SIZE,
    shuffle=True,
    collate_fn=lambda x: process_batch(x, tokenizer, get_gold_path=False)
  )
    
  # for b in train_dataloader: 
  #   print("##############################################")
  #   print(b.tokens)
  #   print(b.confs)
  #   print(b.moves)
  #   print(b.heads)
  
  
  model = BERTNet(device, tokenizer).to(device)
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

  
