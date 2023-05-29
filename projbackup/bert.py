# %%
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from typing import List

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tokenizer=AutoTokenizer.from_pretrained('bert-base-uncased')
tokenizer.add_tokens(["<ROOT>", "<EMPTY>"], special_tokens=True)

print(device)
torch.manual_seed(99)


# %% [markdown]
# # Data
# 


# %%
def is_projective(tree):
    for i in range(len(tree)):
        if tree[i] == -1:
            continue
        left = min(i, tree[i])
        right = max(i, tree[i])

        for j in range(0, left):
            if tree[j] > left and tree[j] < right:
                return False
        for j in range(left + 1, right):
            if tree[j] < left or tree[j] > right:
                return False
        for j in range(right + 1, len(tree)):
            if tree[j] > left and tree[j] < right:
                return False

    return True


def generate_gold_path(sentence:List[str], gold:List[int]) -> tuple[List[List[int]], List[int]]:
  '''
  sentence: list of tokens
  gold: list of heads
  '''
  parser = ArcEager(sentence)
  oracle = Oracle(parser, gold)

  gold_configurations:List[List[int]] = []
  gold_moves:List[int] = []

  while not parser.is_tree_final():
      # save configuration - index of token in sentence
      configuration = [
          parser.stack[ - 1],
      ]
      if len(parser.buffer) == 0:
          configuration.append(-1)
      else:
          configuration.append(parser.buffer[0])
      
      # save configuration    
      gold_configurations.append(configuration)

          # save gold move
      if oracle.is_left_arc_gold():
          gold_moves.append(0)
          parser.left_arc()
      elif oracle.is_right_arc_gold():
          gold_moves.append(1)
          parser.right_arc()
      elif oracle.is_shift_gold():
          gold_moves.append(2)
          parser.shift()
      elif oracle.is_reduce_gold():
          gold_moves.append(3)
          parser.reduce()
  
  return gold_configurations, gold_moves, 

def match_subtokens(l1:List[str], l2:List[str]):
    # Create output list
    output:List[List[int]] = []
    # Initialize index for l2
    index = 0
    # Iterate through l1
    for token in l1:
        subtoken_indices = []
        # Get the indices of the subtokens
        while index < len(l2) and (not subtoken_indices or l2[index].startswith("#")):
            subtoken_indices.append(index)
            index += 1
        # Append subtoken indices to output
        output.append(subtoken_indices)
    return output

def merge_splitted_tokens(tokens:List[str]):
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
        correspondences.append(match_subtokens(t, list(tokenizer.convert_ids_to_tokens(bt))))
    return correspondences


def get_configurations(toks:List[List[str]], heads:List[List[int]], get_gold_path=False):
  '''
  toks: list of list of tokens
  heads: list of list of heads
  '''
  # put sentence and gold tree in our format
      # gold_path and gold_moves are parallel arrays whose elements refer to parsing steps
  gold_configurations:List[List[List[int]]]= (
      []
  )  # record two topmost stack tokens and first 2 buffer token for current step
  gold_moves:List[List[int]] = (
      []
  )  # contains oracle (canonical) move for current step: 0 is left, 1 right, 2 shift, 3 reduce
  gold_heads:List[List[int]]=[]
  
  for tokens, head in zip(toks, heads):
      conf=[]   
      mov=[]

      tokens = ["<ROOT>"] + tokens
      head = [-1] + list(map(int,head))

      if get_gold_path:  # only for training
          conf, mov=generate_gold_pathmoves(tokens, head)
          
          
      gold_configurations.append(conf)
      gold_moves.append(mov)
      gold_heads.append(head)

  return gold_configurations, gold_moves,gold_heads
  

def prepare_batch(batch_data,get_gold_path=False):
    '''
    :param batch_data: batch from dataloader
    :param get_gold_path: if True, return gold path and moves
    
    :return: tokenizer()
    :return: configurations
    :return: moves
    :return: heads
    :return: correspondences
    '''
    # Extract embeddings
    tok_sentences= tokenizer(
        ["<ROOT> "+bd["text"] for bd in batch_data],
        padding=True, 
        return_tensors="pt",
        add_special_tokens=False 
    )

    # get gold path and moves
    configurations, moves, head = get_configurations(
        [bd["tokens"] for bd in batch_data],
        [bd["head"] for bd in batch_data],
        get_gold_path
    )
    
    # get correspondences between tokens and bert tokens (subword)
    correspondences=tokens_tokenizer_correspondence(
        [["<ROOT>"]+bd["tokens"] for bd in batch_data],
        tok_sentences["input_ids"]  # type: ignore 
    )

    return tok_sentences, configurations, moves, head, correspondences





# %%
from datasets import load_dataset

train_dataset=load_dataset("universal_dependencies", "en_lines", split="train")
validation_dataset=load_dataset("universal_dependencies", "en_lines", split="validation")
test_dataset=load_dataset("universal_dependencies", "en_lines", split="test")
# group the following three print in one
print(len(train_dataset)) #type:ignore
print(len(validation_dataset)) #type:ignore
print(len(test_dataset)) #type:ignore

# remove non projective
train_dataset = train_dataset.filter(lambda x:is_projective([-1]+list(map(int,x['head'])))) 
validation_dataset = validation_dataset.filter(lambda x:is_projective([-1]+list(map(int,x['head']))))
test_dataset = test_dataset.filter(lambda x:is_projective([-1]+list(map(int,x['head']))))
print(len(train_dataset)) #type:ignore
print(len(validation_dataset)) #type:ignore
print(len(test_dataset)) #type:ignore



# %% [markdown]
# # NET
# 

# %%
BATCH_SIZE = 32 
DIM_CONFIG = 2
LSTM_ISBI = True
BERT_SIZE = 768
EMBEDDING_SIZE = BERT_SIZE
DIM_CONFIG = 2
LSTM_LAYERS = 1
MLP_SIZE = 200
CLASSES = 4
DROPOUT = 0.2
EPOCHS = 20 # 30
LR = 0.001  # learning rate
NUM_LABELS_OUT = 4
# %%
# ## Dataloader
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
    
    self.w1=nn.Linear(DIM_CONFIG*BERT_SIZE, MLP_SIZE)
    self.w2=nn.Linear(MLP_SIZE, CLASSES)
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

  
  def get_configurations(self, parsers):
    configurations = []
    for parser in parsers:
      if parser.is_tree_final():
        conf = [-1, -1]
      else:
        conf = [
          parser.stack[len(parser.stack) - 1],
        ]
        if len(parser.buffer) == 0:
          conf.append(-1)
        else:
          conf.append(parser.buffer[0])
      configurations.append([conf])
    # print(f"configurations {configurations}")
    return configurations

  
  def infere(self, bertInput):
    bertInput=bertInput.to(self.device)
    input_ids=bertInput['input_ids'].to(self.device)
    attention_mask=bertInput['attention_mask'].to(self.device)
    merged_tokens=[merge_splitted_tokens(list(tokenizer.convert_ids_to_tokens(tok))) for tok in input_ids]

    parsers:List[ArcEager] =[ArcEager(tok) for tok in merged_tokens]
    correspondences= tokens_tokenizer_correspondence(merged_tokens , input_ids)
    h = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state.to(self.device)
    
    
    while not all([t.is_tree_final() for t in parsers]):
      # get the current configuration and score next moves
      configurations = self.get_configurations(parsers)
      mlp_input = self.get_mlp_input(configurations, h, correspondences).to(self.device)
      mlp_out = self.mlp_pass(mlp_input).to(self.device)
      # take the next parsing step
      parse_step(parsers, mlp_out)

    # return the predicted dependency tree
    return [parser.arcs for parser in parsers]
  
      
model = BERTNet(device).to(device)
print(model)



# %%
from utils import evaluate

def train(model:BERTNet, dataloader, criterion, optimizer):
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
        sentences, _, _, trees, _ = batch

        with torch.no_grad():
            pred = model.infere(sentences )
            gold += head
            preds += pred
            count += 1

    return evaluate(gold, preds)

# %%

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

with open("bert.log", "w") as f:
    for epoch in range(EPOCHS):
        print("Starting Epoch", epoch)
        avg_train_loss = train(model, train_dataloader, criterion, optimizer)
        val_uas = test(model, validation_dataloader)

        log= f"Epoch: {epoch:3d} | avg_train_loss: {avg_train_loss:.5f} | dev_uas: {val_uas:.5f} |"
        print(log)
        f.write(log)

        #save the model on pytorch format
        torch.save(model.state_dict(), f"bert_e{epoch+1}.pt")
        #torch.load(f"model_e{epoch}.pt")

    test_uas = test(model, test_dataloader)
    log = "test_uas: {:5.3f}".format(test_uas)
    print(log)
    f.write(log + "\n")

