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
          conf, mov=generate_gold_path(tokens, head)
          
          
      gold_configurations.append(conf)
      gold_moves.append(mov)
      gold_heads.append(head)

  return gold_configurations, gold_moves,gold_heads
  

def prepare_batch(batch_data,get_gold_path=False):
    print(f"batch data type {type(batch_data)}")
    tok_sentences= tokenizer(["<ROOT> "+bd["text"] for bd in batch_data], padding=True, return_tensors="pt", add_special_tokens=False ) # FIXME : add ROOT token
    configurations, moves, gold = get_configurations(
        [bd["tokens"] for bd in batch_data],
        [bd["head"] for bd in batch_data],
        get_gold_path) 
    correspondences=tokens_tokenizer_correspondence(
        [["<ROOT>"]+bd["tokens"] for bd in batch_data],
        tok_sentences["input_ids"])  # type: ignore 

    return tok_sentences, configurations, moves, gold, correspondences




# %%
# processed_sample = tokenizer(train_dataset["text"]) # input_ids token_type_ids attention_mask

# processed_sample.update(get_oracledata(train_dataset["tokens"], train_dataset["head"])) # configurations moves

# processed_sample.keys()

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
# ## Dataloader
# 

# %%
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
from transformers import AutoModel
from arceagerparser import ArcEager 
#modelBert=AutoModel.from_pretrained('bert-base-uncased')

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
  
  def get_mlp_input(self, configs, h, correspondences):
    def getAvgH(h, corr):
      avgH=torch.zeros(BERT_SIZE,requires_grad=False).to(self.device)
      for i in corr:
        avgH+=h[i]
      avgH/=len(corr)
      return avgH
    
    c=0
    mlp_input=[]
    zero_tensor=torch.zeros(BERT_SIZE,requires_grad=False, device=self.device)
    for i, (conf, corr) in enumerate(zip(configs, correspondences)):
      c+=len(conf)
      for j in conf:
        mlp_input.append(
          torch.cat([
            zero_tensor if j[0]==-1 else getAvgH(h[i], corr[j[0]]),
            zero_tensor if j[1]==-1 else getAvgH(h[i], corr[j[1]])
          ])
        )
    mlp_input=torch.stack(mlp_input)
    return mlp_input
  
  
  def forward(self, bertInput, configs, correspondencens):
    # --------------------------------- BERT  ---------------------------------
    #x=[self.dropout(self.embeddings(torch.tensor(s).to(self.device))) for s in bertInput]
    bertInput=bertInput.to(self.device)
    input_ids=bertInput['input_ids'].to(self.device)
    attention_mask=bertInput['attention_mask'].to(self.device)
    
        # Apply the BERT model. This will return a sequence of hidden-states at the output of the last layer of the model.
    outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

    # Get the last hidden state of the token `[CLS]` for each example. BERT gives this as the first token in the sequence.
    h = outputs.last_hidden_state
    h = h.to(self.device)
    

    # Apply dropout on cls_output (not on the input)
    # --------------------------------- LINEAR--------------------------------
    # mlp_input = self.get_mlp_input(configurations, h, correspondences)
    def getAvgH(h, corr):
      avgH=torch.zeros(BERT_SIZE,requires_grad=False).to(self.device)
      for i in corr:
        avgH+=h[i]
      avgH/=len(corr)
      return avgH
    
    c=0
    mlp_input=[]
    zero_tensor=torch.zeros(BERT_SIZE,requires_grad=False).to(self.device)
    for i, (conf, corr) in enumerate(zip(configs, correspondencens)):
      c+=len(conf)
      for j in conf:
        mlp_input.append(
          torch.cat([
            zero_tensor if j[0]==-1 else getAvgH(h[i], corr[j[0]]),
            zero_tensor if j[1]==-1 else getAvgH(h[i], corr[j[1]])
          ])
        )
    mlp_input=torch.stack(mlp_input)
    
    out=self.softmax(self.w2(self.activation(self.w1(self.dropout(mlp_input)))))
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

  def parsed_all(self, parsers):
        for parser in parsers:
            if not parser.is_tree_final():
                return False
        return True

  def merge_splitted_tokens(self, tokens:List[str]):
    out:List[str]=[]
    len=0
    for t in tokens:
      if t[0]=="#":
        out[len-1]+=t.strip("#")
      else:
        out.append(t)
        len+=1
    return out
      
  def mlp_pass(self, x):
      x=x.to(self.device)
      return self.softmax(
          self.w2(self.dropout(self.activation(self.w1(self.dropout(x)))))
      )
  
  def infere(self, bertInput):
    bertInput=bertInput.to(self.device)
    input_ids=bertInput['input_ids'].to(self.device)
    attention_mask=bertInput['attention_mask'].to(self.device)
    merged_tokens=[self.merge_splitted_tokens(list(tokenizer.convert_ids_to_tokens(tok))) for tok in input_ids]

    parsers:List[ArcEager] =[ArcEager(tok) for tok in merged_tokens]
    parsers=parsers
    # pass list of tokens merged and non merged 
    correspondences= tokens_tokenizer_correspondence(merged_tokens , input_ids)
    correspondences=correspondences
    h = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state.to(self.device)
    
    
    while not all([t.is_tree_final() for t in parsers]):# self.parsed_all(parsers):
      # get the current configuration and score next moves
      configurations = self.get_configurations(parsers)
      mlp_input = self.get_mlp_input(configurations, h, correspondences)
      mlp_input = mlp_input.to(self.device)
      mlp_out = self.mlp_pass(mlp_input).to(self.device)
      # take the next parsing step
      self.parse_step(parsers, mlp_out)

    # return the predicted dependency tree
    return [parser.arcs for parser in parsers]
  
  
  
  # In this function we select and perform the next move according to the scores obtained.
  # We need to be careful and select correct moves, e.g. don't do a shift if the buffer
  # is empty or a left arc if σ2 is the ROOT. For clarity sake we didn't implement
  # these checks in the parser so we must do them here. This renders the function quite ugly
  # 0 Lx; 1 Rx, 2 shifr; 3 reduce
  def parse_step(self, parsers, moves):
    moves_argm = moves.argmax(-1)
    for i in range(len(parsers)):
        noMove = False
        # Conditions
        cond_left = (
            len(parsers[i].stack)
            and len(parsers[i].buffer)
            and parsers[i].stack[-1] != 0
        )
        cond_right = len(parsers[i].stack) and len(parsers[i].buffer)
        cond_reduce = len(parsers[i].stack) and parsers[i].stack[-1] != 0
        cond_shift = len(parsers[i].buffer) > 0
        if parsers[i].is_tree_final():
            continue
        else:
            if moves_argm[i] == 0:
#------------------------------ firdt condition to check is the left arc -> right arc -> shift -> reduce------------------------------
                if cond_left:
                    parsers[i].left_arc()
                else:
                    if cond_right:
                        parsers[i].right_arc()
                    elif cond_shift:
                        parsers[i].shift()
                    elif cond_reduce:
                        parsers[i].reduce()
                    else:
                        print("noMove was possible on left")
#------------------------------ firdt condition to check is the right arc -> shift -> reduce------------------------------
            if moves_argm[i] == 1:
                #print("right")
                if cond_right:
                    parsers[i].right_arc()
                else:
                    if cond_shift:
                        parsers[i].shift()
                    elif cond_reduce:
                        parsers[i].reduce() 
                    else:
                        print("noMove was possible on right")
#------------------------------ firdt condition to check is the shift -> reduce------------------------------
            if moves_argm[i] == 2:
                if cond_shift:
                    parsers[i].shift()
                elif cond_reduce:
                    parsers[i].reduce()
                else:
                    print("noMove was possible on shift")
#------------------------------ firdt condition to check is the reduce and if no reduce was possible take in account the probabilities ------------------------------
            if moves_argm[i] == 3:
                if cond_reduce:
                    parsers[i].reduce()
                else:
                    if moves[i][0] > moves[i][1] and moves[i][0] > moves[i][2] and cond_left:
                        parsers[i].left_arc()
                    else:
                        if moves[i][1] > moves[i][2] and cond_right:
                            parsers[i].right_arc()
                        else:
                            if cond_shift:
                                parsers[i].shift()
                            else:
                                print(moves[i][0], moves[i][1], moves[i][2], cond_left, cond_right, cond_shift)
                
      
model = BERTNet(device)
model = model.to(device)


# %% [markdown]
# ## run model
# 

# %%
def train(model:BERTNet, dataloader, criterion, optimizer):
    model.train()  # setup model for training mode
    total_loss = 0
    count = 0
    for batch in dataloader:
        optimizer.zero_grad()
        sentences, paths, moves, trees, correspondences = batch
        out = model(sentences, paths, correspondences)
        ##out = model(input_ids=sentences['input_ids'].to(device), 
        ##    attention_mask=sentences['attention_mask'].to(device), 
        ##    paths)

        labels = torch.tensor(sum(moves, [])).to(
            device
        )  # sum(moves, []) flatten the array
        loss = criterion(out, labels)
        count += 1
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    return total_loss / count

def evaluate(gold:List[List[int]], preds:List[List[int]]):
    total = 0
    correct = 0
    for g, p in zip(gold, preds):
        for i in range(1, len(g)):
            total += 1
            if g[i] == p[i]:
                correct += 1
    return correct / total

def test(model, dataloader:torch.utils.data.DataLoader):
    model.eval()
    gold = []
    preds = []
    for batch in dataloader:
        sentences, paths, moves, trees, correspondences = batch
        with torch.no_grad():
            pred = model.infere(sentences )
            gold += trees
            preds += pred
    return evaluate(gold, preds)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
with open("bert.log", "w") as logbert:
    for epoch in range(EPOCHS):
        print("Starting Epoch", epoch)
        avg_train_loss = train(model, train_dataloader, criterion, optimizer)
        print("AvgTrainLoss", avg_train_loss)
        torch.save(model.state_dict(), f"bert_e{epoch+1}.pt")
        torch.load(f"model_e{epoch}.pt")
        val_uas = test(model, validation_dataloader)
        log= f"Epoch: {epoch:3d} | avg_train_loss: {avg_train_loss:.5f} | dev_uas: {val_uas:.5f} |"
        print(log)
        logbert.write(log)
        #save the model on pytorch format


# %%
