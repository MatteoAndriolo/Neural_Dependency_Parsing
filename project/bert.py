# %%
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from typing import List,Tuple
from utils import is_projective
from arceagerparser import generate_gold
from datasets import load_dataset
from utils import check_bert_format, check_bert_heads, check_bert_sentence


# %% [markdown]
# # Data
# 

# %%
def tok2subt_indices(l1:List[str], l2:List[str]):
    """
    Given two lists of tokens, return a list of lists of indices of l2 that correspond to each token in l1.
    
    Example:
    l1 = ["I", "like", "apples"]
    l2 = ["I", "like", "ap", "##ples"]
    token_corrispondence(l1, l2) -> [[0], [1], [2, 3]]
    
    :param l1: list of tokens
    :param l2: list of subtokens
    :return: list of lists of indices of l2 that correspond to each token in l1
    """
    # Create output list
    output:List[List[int]] = []
    # Initialize index for l2
    index = 0
    # Iterate through l1
    for token in l1:
        subtoken_indices = []
        # Get the indices of the subtokens
        while index < len(l1) and (not subtoken_indices or l1[index].startswith("#")):
            subtoken_indices.append(index)
            index += 1
        # Append subtoken indices to output
        output.append(subtoken_indices)
    return output

def subtok2tok(tokens:List[str]):
    """
    Given a list of subtokens, return a list of tokens.
    
    Example:
    tokens = ["I", "like", "ap", "##ples"]
    subtok2tok(tokens) -> ["I", "like", "apples"]
    
    input:
        tokens: list of subtokens
    returns:
        list of tokens
    """
    out:List[str]=[]
    len=0
    for t in tokens:
      if t[0]=="#":
        out[len-1]+=t.strip("#")
      else:
        out.append(t)
        len+=1
    return out


def tokens_tokenizer_correspondence(tokens:List[List[str]], berttokens:List[List[int]]):
    global tokenizer
    correspondences:List[List[List[int]]]=[]
    
    for t,bt in zip(tokens, berttokens):
        correspondences.append(tok2subt_indices(t, list(tokenizer.convert_ids_to_tokens(bt))))
    return correspondences


def generate_all_golds(toks:List[List[str]], heads:List[List[int]], get_gold_path=False):
  '''
  Generate moves configurations heads for a given parser and oracle
  
  input:
      toks: list of tokens
      heads: list of heads
      
  returns:
      moves: list of moves
      configurations: list of configurations
      arcs: list of heads
  '''
  t:List[Tuple[List, List, List]]= list(map(
    generate_gold,
    [["<ROOT>"]+t for t in toks], 
    [[-1]+h for h in heads],
    ))

  movs, conf, arcs = zip(*t)
  return list(movs), list(conf), list(arcs)

def prepare_batch(batch_data,get_gold_path=False):
    '''
    Prepare batch for NN ingestion
    
    input:
        batch_data: batch from dataloader
        get_gold_path: if True, return gold path and moves
        
    returns:
        tok_sentences: tokenized sentenceshead
        configurations: list of configurations
        moves: list of moveshead
        head: list of heads
        correspondences: list of correspondences between tokens and bert tokens (subword)
    '''
    # Extract embeddingshead
    tok_sentences= tokenizer(
        ["<ROOT> "+bd["text"] for bd in batch_data],
        padding=True, 
        return_tensors="pt",
        add_special_tokens=False 
    )

    # get gold path and moves
    moves, configurations, head = generate_all_golds(
        [bd["tokens"] for bd in batch_data],
        [bd["head"] for bd in batch_data],
        get_gold_path
    )

    # get correspondences between tokens and bert tokens (subword)
    correspondences=list(
      map(tok2subt_indices ,
          [["<ROOT>"]+bd["tokens"] for bd in batch_data],
          tok_sentences["input_ids"]  # type: ignore 
      )
    )

    return tok_sentences, configurations, moves, head, correspondences


# %% [markdown]
# # NET
# 

# %%
BATCH_SIZE = 256
DIM_CONFIG = 2
BERT_SIZE = 768
EMBEDDING_SIZE = BERT_SIZE
MLP_SIZE = 200
CLASSES = 4
DROPOUT = 0.2
EPOCHS = 20 #
LR = 0.001  # learning rate
NUM_LABELS_OUT = 4
# %% [markdown]
# ## Dataloader

#%%

from transformers import AutoModel
from arceagerparser import ArcEager, Oracle
from utils import parse_step

class BERTNet(nn.Module):
  def __init__(self,device) -> None:
    super().__init__()
    self.device=device
    
    self.embeddings = nn.Embedding(
        len(tokenizer), EMBEDDING_SIZE, padding_idx=0
    )
    
    self.bert = AutoModel.from_pretrained('bert-base-uncased')
    self.bert.resize_token_embeddings(len(tokenizer))
    for param in self.bert.parameters():
      param.requires_grad = False
    
    self.w1=nn.Linear(DIM_CONFIG*BERT_SIZE, 3*MLP_SIZE )
    self.w2=nn.Linear(3*MLP_SIZE, CLASSES)
    self.activation= nn.Tanh()
    self.softmax=nn.Softmax(dim=-1)
    self.dropout = nn.Dropout(DROPOUT)
  

  def getAvgH(self, h, corr):
    avgH=torch.zeros(BERT_SIZE,requires_grad=False).to(self.device)
    for i in corr:
        avgH+=h[i]
    avgH/=len(corr)
    return avgH
  
  def get_mlp_input(self, configs, h, correspondences):
    mlp_input=[]
    zero_tensor=torch.zeros(BERT_SIZE,requires_grad=False, device=self.device)
    for i, (conf, corr) in enumerate(zip(configs, correspondences)):
      for j in conf:
        mlp_input.append(
          torch.cat([
            zero_tensor if j[0]==-1 else self.getAvgH(h[i], corr[j[0]]),
            zero_tensor if j[1]==-1 else self.getAvgH(h[i], corr[j[1]])
          ])
        )
    mlp_input=torch.stack(mlp_input).to(self.device)
    return mlp_input
  
  def mlp_pass(self, x):
      return self.softmax(
          self.w2(self.dropout(self.activation(self.w1(self.dropout(x)))))
      )

  
  def forward(self, bertInput, configs, correspondencens):
    bertInput=bertInput.to(self.device)
    input_ids=bertInput['input_ids'].to(self.device)
    attention_mask=bertInput['attention_mask'].to(self.device)
    
    h = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state.to(self.device)

    mlp_input=self.get_mlp_input(configs, h, correspondencens)
    out=self.mlp_pass(mlp_input)

    return out

  
  def infere(self, bertInput):
    bertInput=bertInput.to(self.device)
    input_ids=bertInput['input_ids'].to(self.device)
    attention_mask=bertInput['attention_mask'].to(self.device)
    merged_tokens=[subtok2tok(list(tokenizer.convert_ids_to_tokens(tok))) for tok in input_ids]

    parsers:List[ArcEager] =[ArcEager(tok) for tok in merged_tokens]
    correspondences= list(map(
      tok2subt_indices,
      merged_tokens,
      list(map(list, map(tokenizer.convert_ids_to_tokens, input_ids)))
    ))

    h = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state.to(self.device)
    
    while not all([t.is_tree_final() for t in parsers]):
      # get the current configuration and score next moves
      configurations = [p.get_configuration_now() for p in parsers]
      mlp_input = self.get_mlp_input(configurations, h, correspondences).to(self.device)
      mlp_out = self.mlp_pass(mlp_input).to(self.device)
      # take the next parsing step
      parse_step(parsers, mlp_out)

    # return the predicted dependency tree
    return [p.list_arcs for p in parsers]
  
      
# %% [markdown]
# ## Run the model

# %%
from utils import evaluate

def train(model:BERTNet, dataloader:torch.utils.data.DataLoader, criterion, optimizer): #type:ignore
    model.train()  # setup model for training mode
    total_loss = 0
    count = 0
    
    for batch in dataloader:
        print(f"TRAINING: batch {count}/{len(dataloader):.0f}")
        optimizer.zero_grad()
        sentences, paths, moves, _, correspondences = batch

        out = model(sentences, paths, correspondences)

        labels = torch.tensor(sum(moves, [])).to(
            device
        )  # sum(moves, []) flatten the array

        loss = criterion(out, labels)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        count += 1
    return total_loss / count


def test(model:BERTNet, dataloader:torch.utils.data.DataLoader): #type:ignore
    model.eval()
    
    gold = []
    preds = []
    count = 0
    for batch in dataloader:
        print(f"TEST: batch {count}/{len(dataloader):.0f}")
        sentences, _ , _ , head, _ = batch
        
        with torch.no_grad():
            pred = model.infere(sentences )
            gold += head
            preds += pred
            count += 1

    return evaluate(gold, preds)
# %%
if __name__=="__main__":
  # torch settings
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print(device)
  torch.manual_seed(99)

  # tokenizer settings
  tokenizer=AutoTokenizer.from_pretrained('bert-base-uncased')
  tokenizer.add_tokens(["<ROOT>", "<EMPTY>"], special_tokens=True)

  ## DATA
  train_dataset=load_dataset("universal_dependencies", "en_lines", split="train")
  validation_dataset=load_dataset("universal_dependencies", "en_lines", split="validation")
  test_dataset=load_dataset("universal_dependencies", "en_lines", split="test")
  print(
      f"train_dataset: {len(train_dataset)}, validation_dataset: {len(validation_dataset)}, test_dataset: {len(test_dataset)}" #type:ignore
  ) 

  # remove non projective
  train_dataset = train_dataset.filter(lambda x:is_projective([-1]+list(map(int,x['head'])))) 
  #validation_dataset = validation_dataset.filter(lambda x:is_projective([-1]+list(map(int,x['head']))))
  #test_dataset = test_dataset.filter(lambda x:is_projective([-1]+list(map(int,x['head']))))
  print(
      f"PROJECTIVE -> train_dataset: {len(train_dataset)}, validation_dataset: {len(validation_dataset)}, test_dataset: {len(test_dataset)}" #type:ignore
  )   
  
  train_dataloader = torch.utils.data.DataLoader( # type:ignore
    train_dataset,
    batch_size=BATCH_SIZE, 
    shuffle=True,
    collate_fn=lambda x: prepare_batch(x, get_gold_path=True)
  )
  validation_dataloader = torch.utils.data.DataLoader( # type: ignore
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=lambda x: prepare_batch(x, get_gold_path=True)
  )

  test_dataloader = torch.utils.data.DataLoader( # type:ignore
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=lambda x: prepare_batch(x, get_gold_path=False)
  )

  # %%
  model = BERTNet(device)
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=LR)

  with open("bert.log", "w") as f:
      for epoch in range(EPOCHS):
          print("Starting Epoch", epoch)
          avg_train_loss = train(model, train_dataloader, criterion, optimizer)
          torch.save(model.state_dict(), "bert.pt")
          #avg_train_loss = -1
          #torch.load(f"bert.pt")
          val_uas = test(model, validation_dataloader)

          log= f"Epoch: {epoch:3d} | avg_train_loss: {avg_train_loss:.5f} | dev_uas: {val_uas:.5f} |\n"
          print(log)
          f.write(log)

          #save the model on pytorch format

      test_uas = test(model, test_dataloader)
      log = f"test_uas: {test_uas:5.3f}"
      print(log)
      f.write(log + "\n")
      
      
    # from datasets import load_dataset
    # from utils import is_projective
    # errors=False
    # training_dataset=load_dataset("universal_dependencies", "en_lines", split="train")
    # training_dataset=training_dataset.filter(lambda x: is_projective([-1]+list(map(int,x["head"]))))
    # from bert import generate_all_golds
    # configurations, moves, head=generate_all_golds([db["tokens"] for db in training_dataset], [db["head"] for db in training_dataset])

    # for h,a in zip(head, [bd["head"] for bd in training_dataset]):
    #     if  a != h:
    #         print(f"ERROR HEADS ")
    #         errors=True
    #         break
    
    # if errors:
    #     print("ERRORS FOUND")
    #     print("TEST NOT PASSED")
    # else:
    #     print("TEST PASSED")
        
# %%
