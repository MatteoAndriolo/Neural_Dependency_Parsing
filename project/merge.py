# %%
import torch
import torch.nn as nn
from torch import sort as tsort, Tensor
import time
from typing import List, Tuple
from transformers import AutoTokenizer, AutoModel
from transformers.tokenization_utils_base import BatchEncoding
from datasets import load_dataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
torch.manual_seed(99)
# %% [markdown]
# Arc-Eager Parser and Oracle


# Arc Eager Parser


# %%
from typing import List

NOMOVE = -1
LEFT_ARC = 0
RIGHT_ARC = 1
REDUCE = 2
SHIFT = 3

IS_FINAL = -10
EMPTY = -1

class ArcEager:
    def __init__(self, sentence):
        """
        input:
            sentence: list of words | first word must be <ROOT>
            debug: if True print each move
        """
        if all([isinstance(x,str) for x in sentence]):
            if sentence[0] != "<ROOT>":
                raise Exception("ERROR: first word must be <ROOT>")
        elif all([isinstance(x,int) for x in sentence]):
            if sentence[0] != 1: # token of ROOT is 1
                raise Exception("ERROR: first word must be -1")
        else:
            raise Exception("ERROR: sentence must be list of words or list of ints")
            
        self.sentence = sentence
        self.buffer = [i for i in range(len(self.sentence))]
        self.stack = []

        self.list_arcs = [-1 for _ in range(len(self.sentence))]
        self.list_moves=[]
        self.list_configurations = []

        # Do first shift -> add ROOT to stack
        self.stack.append(self.buffer.pop(0))
        self.is_finished=False

    def update_configurations(self, move):
        ''' to do before each move '''
        if move == NOMOVE:
            self.list_configurations.append([EMPTY, EMPTY])
            self.list_moves.append(NOMOVE)
        if len(self.stack)>0:
            self.list_configurations.append([
                self.stack[-1],
                self.buffer[0] if len(self.buffer)>0 else EMPTY
            ])
            self.list_moves.append(move)
            
        
    def left_arc(self):
        self.update_configurations(LEFT_ARC)
        s1 = self.stack.pop(-1)
        b1 = self.buffer[0]
        self.list_arcs[s1] = b1

    def right_arc(self):
        if not is_right_possible(self):
            self.nomove()
            return
        self.update_configurations(RIGHT_ARC)
        s1 = self.stack[-1]
        b1 = self.buffer.pop(0)
        self.stack.append(b1)
        self.list_arcs[b1] = s1

    def shift(self):
        self.update_configurations(SHIFT)
        self.stack.append(self.buffer.pop(0))

    def reduce(self):
        self.update_configurations(REDUCE)
        self.stack.pop()

    def nomove(self):
        self.is_finished=True
        self.update_configurations(NOMOVE)

    def do_move(self, move:int):
        if move==LEFT_ARC: 
            self.left_arc()
        elif move==RIGHT_ARC:
            self.right_arc()
        elif move==SHIFT:
            self.shift()
        elif move==REDUCE:
            self.reduce()
        elif move==NOMOVE:
            self.nomove()
        return move

    def is_tree_final(self):
        return self.is_finished or (len(self.stack) == 1 and len(self.buffer) == 0) 

    def print_configuration(self):
        s = [self.sentence[i] for i in self.stack]
        b = [self.sentence[i] for i in self.buffer]
        print(s, b)
        print(self.stack, self.buffer)
        print(self.list_arcs)

    def get_list_moves(self):
        return self.list_moves

    def get_list_configurations(self):
        return self.list_configurations

    def get_list_arcs(self):
        return self.list_arcs
    
    def get_configuration_now(self):
        if self.is_tree_final():
            conf=[-1,-1]
        else:
            conf=[self.stack[-1]]
            if len(self.buffer) == 0:
                conf.append(-1)
            else:
                conf.append(self.buffer[0])
        return conf
        

# %% [markdown]
## Oracle

# %%
class Oracle:
    def __init__(self, parser, gold_tree:List[int]):
        self.parser = parser
        self.gold = list(map(int,gold_tree))

        # Check correctness of input
        if self.gold[0] != -1:
            print("ERROR: gold tree must start with -1")
            exit(-1)

    """
    i: top of stack, j: top of buffer
    if there's a link j -> i then return LEFT-ARC
    else if there's a link i -> j then return RIGHT-ARC
    else if there's a link k <-/-> j, k < i then return REDUCE
    else return SHIFT 
    """
    def is_left_arc_gold(self):
        # first element of the of the buffer is the gold head of the topmost element of the stack
        # if empty lists or if top has no head -> return False
        if (
            len(self.parser.buffer) == 0
            or self.parser.stack[-1] == 0  # if top is ROOT
        ):
            return False

        s = self.parser.stack[-1]
        b = self.parser.buffer[0]  # [0]
        if self.gold[s] != b:
            return False

        return True

    def is_right_arc_gold(self):
        # if topmost stack element is gold head of the first element of the buffer
        if len(self.parser.buffer) == 0:
            return False

        s = self.parser.stack[-1]
        b = self.parser.buffer[0]  # [0]
        if self.gold[b] != s:
            return False 

        return True 

    def is_reduce_gold(self):
        s = self.parser.stack[-1]
        if self.parser.list_arcs[s] == -1 or s==0: # if top has no head or if top is ROOT
            return False
        if len(self.parser.buffer) == 0:                    # if buffer is empty
            if self.parser.list_arcs[s] != -1 and s != 0:   # if top has a head and top is not ROOT
                return True
            return False

        for i in range(0, len(self.parser.buffer)):
            b = self.parser.buffer[i]
            if self.gold[b] == s or self.gold[s] == b: # if there's a link k <-/-> j, k < i then do not reduce
                return False 
            
        return True 

    def is_shift_gold(self):
        if len(self.parser.buffer) == 0:
            return False
        if self.is_left_arc_gold() or self.is_right_arc_gold() or self.is_reduce_gold():
            return False
        return True
    
    def get_next_move(self, do_it=False):
        if self.parser.is_tree_final():
            return IS_FINAL
        if self.is_left_arc_gold():
            return LEFT_ARC
        elif self.is_right_arc_gold():
            return RIGHT_ARC
        elif self.is_reduce_gold():
            return REDUCE
        elif self.is_shift_gold():
            return SHIFT
        else:
            print("NO MOVE")
            print(self.gold)
            print(self.parser.list_arcs)
            self.parser.print_configuration()
            exit(-5)
            return None


# %%

def is_left_possible(parser):
    return len(parser.stack) >= 1 and len(parser.buffer) >= 1 and parser.stack[-1] != 0

def is_right_possible(parser):
    return len(parser.stack) >= 1 and len(parser.buffer) >= 1

def is_shift_possible(parser):
    return len(parser.buffer) >= 1

def is_reduce_possible(parser):
    return len(parser.stack) >= 1 and parser.list_arcs[parser.stack[-1]] != -1

# %% [markdown]
#Parser used for output of the model in order to accept or reject moves (feasible or unfeasible)

# %%
def parse_moves(parsers:List[ArcEager], moves:Tensor):
    _, indices = tsort(moves, descending=True)
    list_moves=[]
    for i in range(len(parsers)):
        noMove =True 
        if parsers[i].is_tree_final():
           list_moves.append(NOMOVE) 
           continue
        else:
            for j in range(4):
                if indices[i][j] == LEFT_ARC and is_left_possible(parsers[i]):
                    list_moves.append(LEFT_ARC)
                    noMove = False;break;
                elif indices[i][j] == RIGHT_ARC and is_right_possible(parsers[i]):
                    list_moves.append(RIGHT_ARC)
                    noMove = False;break;
                elif indices[i][j] == REDUCE and is_reduce_possible(parsers[i]):
                    list_moves.append(REDUCE)
                    noMove = False;break;
                elif indices[i][j] == SHIFT and is_shift_possible(parsers[i]) :
                    list_moves.append(SHIFT)
                    noMove = False;break;
        if noMove:
            list_moves.append(NOMOVE)
    return list_moves

def generate_gold(sentence:List[str], gold:List[int]):
    '''
    Generate moves configurations heads for a given parser and oracle
    
    input:
        parser: ArcEager object
        oracle: Oracle object
    returns:
        moves: list of moves
        configurations: list of configurations
        arcs: list of heads
        
    '''
    parser:ArcEager=ArcEager(sentence)
    oracle:Oracle=Oracle(parser, gold)

    while not parser.is_tree_final():
        if parser.do_move(oracle.get_next_move()) == NOMOVE:
            print("ERROR: NOMOVE")
        
    return parser.list_moves, parser.list_configurations,  parser.list_arcs
# %% [markdown]
# Data

#%%
def is_projective(head):
    for i in range(len(head)):
        if head[i] == -1:
            continue
        left = min(i, head[i])
        right = max(i, head[i])

        for j in range(0, left):
            if head[j] > left and head[j] < right:
                return False
        for j in range(left + 1, right):
            if head[j] < left or head[j] > right:
                return False
        for j in range(right + 1, len(head)):
            if head[j] > left and head[j] < right:
                return False

    return True
  


# %% [markdown]
## Download data

# %% [markdown]
#Define batchsize
# %%
BATCH_SIZE=256

# %% [markdown]
# Download data
# %%
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

# %% [markdown]
# BiLSTM

# %%

class NNData():
  def __init__(self, tokens, confs, moves, heads) -> None:
      self.enc_tokens = tokens
      self.confs = confs
      self.moves = moves
      self.heads = heads
      #self.dictionary = dictionary

def extract_att(data:List[NNData], attribute:str):
  return [getattr(d, attribute) for d in data]
      

def create_dictionary(dataset, threshold: int =3) -> dict[str, int]:
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

def train(model: nn.Module, dataloader, criterion, optimizer):
  model.train()
  total_loss = 0
  count = 1
  for batch in dataloader:
    print(f"TRAIN: batch {count}/{len(dataloader):.0f}")
    optimizer.zero_grad()
    
    out = model(batch)
    if isinstance(batch[0], NNData):
        moves= extract_att(batch, "moves")
    else:
        moves=extract_att(batch[1],"moves")

    labels = torch.tensor(sum(moves, [])).to(
        device
    )  # sum(moves, []) flatten the array
    
    loss = criterion(out, labels)
    total_loss += loss.item()
    loss.backward()
    optimizer.step()
    count += 1

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

def test(model, dataloader: torch.utils.data.dataloader):  # type:ignore
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
# Network definition

# %%
class NNParameters(): 
    def __init__(self) -> None:
      self.BATCH_SIZE = BATCH_SIZE
      self.EMBEDDING_SIZE = 200
      self.FREEZE = True
      self.LSTM_SIZE = 200
      self.LSTM_LAYERS = 2

      self.MLP_OUT_SIZE = self.LSTM_LAYERS * self.LSTM_SIZE
      self.OUT_CLASSES = 4
      
      self.DROP_OUT = 0.2
      self.LR = 0.001
      self.EPOCHS = 1

nnp = NNParameters()

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
    
    self.w1 = nn.Linear(2 * nnp.LSTM_LAYERS * nnp.LSTM_SIZE , 100  , bias=True)
    self.activation = nn.Tanh()
    self.w2 = nn.Linear(100, nnp.OUT_CLASSES, bias=True)
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
# Dataloader

# %%
dictionary= create_dictionary(train_dataset)
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

# %% [markdown]
# DO SOMETHING !

# %%
model = BiLSTMNet(device, dictionary).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=nnp.LR)

for epoch in range(nnp.EPOCHS):
  print(f"Starting Epoch {epoch+1}/{nnp.EPOCHS}")
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

# %% [markdown]
# BERT

# %%
class NNParameters():
  def __init__(self) -> None:
      self.BATCH_SIZE = BATCH_SIZE 
      self.BERT_SIZE = 768
      self.EMBEDDING_SIZE = self.BERT_SIZE
      self.DIM_CONFIG = 2
      self.MLP1_IN_SIZE = self.DIM_CONFIG * self.EMBEDDING_SIZE
      self.MLP2_IN_SIZE = 300
      self.OUT_CLASSES = 4
      self.FREEZE = True
      self.DROP_OUT = 0.2
      self.LR = 0.01
      self.EPOCHS = 1

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


#%%
tokenizer=AutoTokenizer.from_pretrained('bert-base-uncased')
tokenizer.add_tokens(["<ROOT>", "<EMPTY>"], special_tokens=True)

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
  output_tokenizer= tokenizer(
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
    if nnp.FREEZE:
      for param in self.bert.parameters():
        param.requires_grad = False
    
    self.w1 = nn.Linear(1536, 300, bias=True)
    self.activation = nn.Tanh()
    self.w2 = nn.Linear(300, 4 ,bias=True)
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
# Dataloader

# %%
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

# %% [markdown]
# DO SOMETHING !

# %%
model = BERTNet(device, tokenizer).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=nnp.LR)

for epoch in range(nnp.EPOCHS):
  print("Starting Epoch", epoch)
  # torch.load(f"bilstm_e{epoch+1}.pt")
  avg_train_loss = train(model, train_dataloader, criterion, optimizer)
  if epoch % 5 == 0:
    val_uas = test(model, validation_dataloader)
  else:
    val_uas = -1
    

  log = f"Epoch: {epoch:3d} | avg_train_loss: {avg_train_loss:5.3f} | dev_uas: {val_uas:5.3f} |"
  print(log)

  # save the model on pytorch format
  torch.save(model.state_dict(), f"bilstm_e{epoch+1}.pt")

test_uas = test(model, test_dataloader)
log = "test_uas: {:5.3f}".format(test_uas)
print(log)
train(model, train_dataloader, criterion, optimizer)

print("--- %s seconds ---" % (time.time() - start_time))




# %%
