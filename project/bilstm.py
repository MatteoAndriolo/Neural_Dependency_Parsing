# %%
import torch
import torch.nn as nn
from typing import List

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
torch.manual_seed(99)
BATCH_SIZE=32


# %% [markdown]
# # Data
# 

# %% 
from utils import generate_gold_pathmoves, is_projective

def create_dictionary(dataset, threshold=3):
    '''
    :param dataset: list of list of tokens samples
    :param threshold: minimum number of appearances of a word in the dataset
    :return: a dictionary of word/index pairs
    '''
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

def process_sample(tokens, emb_dictionary, get_gold_path=False):
    '''
    Process a sample from the dataset 
    :param         tokens: tokens of a sentence
    :param emb_dictionary: dictionary of word/index pairs
    :param  get_gold_path: if True, we also return the gold path and gold moves
    :return: enc_sentence: encoded tokens of the sentence
                gold_path: gold path of the sentence
               gold_moves: gold moves of the sentence
                     gold: gold heads of the sentence
    '''
    sentence = ["<ROOT>"] + tokens["tokens"]
    head = [-1] + list(map(int, tokens["head"])) #[int(i) for i in tokens["head"]]  

    # embedding ids of sentence words
    enc_sentence = [
        emb_dictionary[word] if word in emb_dictionary else emb_dictionary["<unk>"]
        for word in sentence
    ]

    gold_path, gold_moves= [], [] if not get_gold_path else generate_gold_pathmoves(sentence, head)

    return enc_sentence, gold_path, gold_moves, head


def prepare_batch(batch_data, get_gold_path=False):
    '''
    :param batch_data: batch from dataloader
    :param get_gold_path: if True, we also return the gold path and gold moves
    
    :return: sentences: list of encoded sentences
    :return     paths: list of gold paths
    :return     moves: list of gold moves
    :return     heads: list of gold heads
    '''
    data = [
        process_sample(s, emb_dictionary, get_gold_path=get_gold_path)
        for s in batch_data
    ]
    # sentences, paths, moves, trees are parallel arrays, each element refers to a sentence
    sentences = [s[0] for s in data]
    paths = [s[1] for s in data]
    moves = [s[2] for s in data]
    heads = [s[3] for s in data]
    # print(f"sentences{len(sentences[0])}")
    # print(f"paths{len(paths[0])}")
    # print("moves", len(moves[0]))
    # print("trees", len(trees[0]))
    return sentences, paths, moves, heads



# %% [markdown]
# ## Prepare dataset
#

# %%
from datasets import load_dataset

train_dataset = load_dataset("universal_dependencies", "en_lines", split="train")
validation_dataset = load_dataset("universal_dependencies", "en_lines", split="validation")
test_dataset = load_dataset("universal_dependencies", "en_lines", split="test")
# print length of datasets
print(f"train_dataset: {len(train_dataset)}, validation_dataset: {len(validation_dataset)}, test_dataset: {len(test_dataset)}") #type:ignore

train_dataset = train_dataset.filter(lambda x:is_projective([-1]+list(map(int,x['head'])))) 
validation_dataset = validation_dataset.filter(lambda x:is_projective([-1]+list(map(int,x['head']))))
test_dataset = test_dataset.filter(lambda x:is_projective([-1]+list(map(int,x['head']))))
print(f"PROJECTIVE -> train_dataset: {len(train_dataset)}, validation_dataset: {len(validation_dataset)}, test_dataset: {len(test_dataset)}") #type:ignore


emb_dictionary = create_dictionary(train_dataset)
print(f"len(emb_dictionary): {len(emb_dictionary)}")
# %% [markdown]
# ## DataLoaders
#

# %%
BATCH_SIZE = 32 #GPU at 80% with 32 batch size

train_dataloader = torch.utils.data.DataLoader(  # type:ignore
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=lambda x: prepare_batch(x, get_gold_path=True),
)
dev_dataloader = torch.utils.data.DataLoader(  # type: ignore
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=lambda x: prepare_batch(x, get_gold_path=True),
)
test_dataloader = torch.utils.data.DataLoader(  # type:ignore
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=lambda x: prepare_batch(x, get_gold_path=False),
)

# %% [markdown]
# # BiLSTM
#

# %%
EMBEDDING_SIZE = 200
LSTM_SIZE = 200
LSTM_LAYERS = 2  # It was 1 before
MLP_SIZE = 200
CLASSES = 4
DROPOUT = 0.2
EPOCHS = 30
LR = 0.001  # learning rate

from arceagerparser import ArcEager, Oracle
class BiLSTMNet(nn.Module):
    def __init__(self, device):
        super(BiLSTMNet, self).__init__()
        self.device = device
        self.embeddings = nn.Embedding(
            len(emb_dictionary), EMBEDDING_SIZE, padding_idx=emb_dictionary["<pad>"]
        )
        self.weight = torch.nn.Parameter(torch.Tensor(400, 1200))

        # initialize bi-LSTM
        self.lstm = nn.LSTM(
            EMBEDDING_SIZE,
            LSTM_SIZE,
            num_layers=LSTM_LAYERS,
            bidirectional=True,
            dropout=DROPOUT,
        )

        # initialize feedforward
        self.w1 = torch.nn.Linear(6 * LSTM_SIZE, MLP_SIZE, bias=True)
        self.activation = torch.nn.Tanh()
        self.w2 = torch.nn.Linear(MLP_SIZE, CLASSES, bias=True)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.dropout = torch.nn.Dropout(DROPOUT)

    def forward(self, x, paths):
        # get the embeddings
        x = [self.dropout(self.embeddings(torch.tensor(i).to(self.device))) for i in x]

        # run the bi-lstm
        # h = self.lstm_pass(x)
        x = torch.nn.utils.rnn.pack_sequence(x, enforce_sorted=False)
        h, (h_0, c_0) = self.lstm(x)
        h, h_sizes = torch.nn.utils.rnn.pad_packed_sequence(h)

        # print(f"h_sizes: {sum(h_sizes)},\n h.shape {h.shape}\n ")

        # for each parser configuration that we need to score we arrange from the
        # output of the bi-lstm the correct input for the feedforward
        # mlp_input = self.get_mlp_input(paths, h)
        mlp_input = []
        zero_tensor = torch.zeros(2 * LSTM_SIZE, requires_grad=False).to(self.device)
        c = 0
        for i in range(len(paths)):
            c += len(paths[i])
            for j in paths[i]:
                mlp_input.append(
                    torch.cat(
                        [
                            zero_tensor if j[0] == -1 else h[j[0]][i],
                            zero_tensor if j[1] == -1 else h[j[1]][i],
                            zero_tensor if j[2] == -1 else h[j[2]][i],
                        ]
                    )
                )
        mlp_input = torch.stack(mlp_input).to(self.device)

        # print(f"len(paths){len(paths)}")
        # run the feedforward and get the scores for each possible action
        # out = self.mlp(mlp_input)
        # print("0", mlp_input.shape)
        x = self.dropout(mlp_input)
        # print("1", x.shape)
        x = self.w1(x)
        # print("2", x.shape)
        x = self.activation(x)
        # print("3", x.shape)
        x = self.dropout(x)
        x = self.w2(x)
        out = self.softmax(x)

        return out

    def lstm_pass(self, x):
        x = torch.nn.utils.rnn.pack_sequence(x, enforce_sorted=False)
        h, (h_0, c_0) = self.lstm(x)
        h, h_sizes = torch.nn.utils.rnn.pad_packed_sequence(h)
        # size h: (length_sentences, batch, output_hidden_units)
        return h

    def get_mlp_input(self, configurations, h):
        mlp_input = []
        zero_tensor = torch.zeros(2 * LSTM_SIZE, requires_grad=False).to(self.device)
        for i in range(len(configurations)):
            # print(f"len of the configurations {len(configurations)}")  # for every sentence in the batch
            for j in configurations[i]:  # for each configuration of a sentence
                # print(f"len of the configurations {configurations[i]}")
                # print(f"len of the configurations {len(configurations[i])}")
                mlp_input.append(
                    torch.cat(
                        [
                            zero_tensor if j[0] == -1 else h[j[0]][i],
                            zero_tensor if j[1] == -1 else h[j[1]][i],
                            zero_tensor if j[2] == -1 else h[j[2]][i],
                        ]
                    )
                )
        mlp_input = torch.stack(mlp_input).to(self.device)
        return mlp_input

    def mlp(self, x):
        return self.softmax(
            self.w2(self.dropout(self.activation(self.w1(self.dropout(x)))))
        )

    # we use this function at inference time. We run the parser and at each step
    # we pick as next move the one with the highest score assigned by the model
    def infere(self, x):
        parsers = [ArcEager(i) for i in x]

        x = [self.embeddings(torch.tensor(i).to(self.device)) for i in x]

        h = self.lstm_pass(x)

        while not self.parsed_all(parsers):
            # get the current configuration and score next moves
            configurations = self.get_configurations(parsers)
            mlp_input = self.get_mlp_input(configurations, h)
            mlp_out = self.mlp(mlp_input)
            # take the next parsing step
            self.parse_step(parsers, mlp_out)

        # return the predicted dependency tree
        return [parser.arcs for parser in parsers]

    def get_configurations(self, parsers):
        configurations = []

        for parser in parsers:
            if parser.is_tree_final():
                conf = [-1, -1, -1]
            else:
                conf = [
                    parser.stack[len(parser.stack) - 2],
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

    # In this function we select and perform the next move according to the scores obtained.
    # We need to be careful and select correct moves, e.g. don't do a shift if the buffer
    # is empty or a left arc if σ2 is the ROOT. For clarity sake we didn't implement
    # these checks in the parser so we must do them here. This renders the function quite ugly
    # 0 Lx; 1 Rx, 2 shifr; 3 reduce
    def parse_step(self, parsers: List[ArcEager], moves):
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
                    



model = BiLSTMNet(device).to(device)
print(model)




# %%
from utils import evaluate


def train(model:BiLSTMNet, dataloader, criterion, optimizer):
    model.train()  # setup model for training mode
    total_loss = 0
    count = 0

    for batch in dataloader:
        print(f"TRAINING: batch {count}/{len(dataloader)/BATCH_SIZE:.0f}")
        optimizer.zero_grad()
        sentences, paths, moves, _ = batch

        out = model(sentences, paths)

        labels = torch.tensor(sum(moves, [])).to(
            device
        )  # sum(moves, []) flatten the array

        loss = criterion(out, labels)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        count += 1

    return total_loss / count


def test(model:BiLSTMNet, dataloader:torch.utils.data.DataLoader): #type:ignore
    model.eval()

    gold = []
    preds = []

    for batch in dataloader:
        sentences, _, _, head = batch

        with torch.no_grad():
            pred = model.infere(sentences)
            gold += head
            preds += pred

    return evaluate(gold, preds)


# %%

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)


with open("bilstm.log", "w") as f:
    for epoch in range(EPOCHS):
        print("Starting Epoch", epoch)
        avg_train_loss = train(model, train_dataloader, criterion, optimizer)
        val_uas = test(model, dev_dataloader)

        log=f"Epoch: {epoch:3d} | avg_train_loss: {avg_train_loss:5.3f} | dev_uas: {val_uas:5.3f} |"
        print(log)
        f.write(log+"\n")
        #save the model on pytorch format
        torch.save(model.state_dict(), f"bilstm_e{epoch+1}.pt")
        #torch.load(f"bilstm_e{epoch+1}.pt")

    test_uas = test(model, test_dataloader)
    log="test_uas: {:5.3f}".format(test_uas)
    print(log)
    f.write(log+"\n")
        
