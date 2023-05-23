# %%
import torch
import torch.nn as nn
import torch.utils as tutils
import pandas as pd
from typing import List

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
torch.manual_seed(99)

# %% [markdown]
# # Dataset preparation and DataLoader
#

# %% [markdown]
# ## Function definitions
#


# %%
# the function returns whether a tree is projective or not. It is currently
# implemented inefficiently by brute checking every pair of arcs.
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


# the function creates a dictionary of word/index pairs: our embeddings vocabulary
# threshold is the minimum number of appearance for a token to be included in the embedding list
def create_dictionary(dataset, threshold=3):
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


# %%
from arceagerparser import ArcEager, Oracle


def process_sample(sample, emb_dictionary, get_gold_path=False):
    # put sentence and gold tree in our format
    sentence = ["<ROOT>"] + sample["tokens"]
    gold = [-1] + [
        int(i) for i in sample["head"]
    ]  # heads in the gold tree are strings, we convert them to int

    # embedding ids of sentence words
    enc_sentence = [
        emb_dictionary[word] if word in emb_dictionary else emb_dictionary["<unk>"]
        for word in sentence
    ]

    # gold_path and gold_moves are parallel arrays whose elements refer to parsing steps
    gold_path = (
        []
    )  # record two topmost stack tokens and first buffer token for current step
    gold_moves = (
        []
    )  # contains oracle (canonical) move for current step: 0 is left, 1 right, 2 shift, 3 reduce

    if get_gold_path:  # only for training
        parser = ArcEager(sentence)
        oracle = Oracle(parser, gold)

        while not parser.is_tree_final():
            # save configuration
            configuration = [
                parser.stack[len(parser.stack) - 2],
                parser.stack[len(parser.stack) - 1],
            ]
            if len(parser.buffer) == 0:
                configuration.append(-1)
            else:
                configuration.append(parser.buffer[0])
            gold_path.append(configuration)

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

    # print("enc_sentence", len(enc_sentence))
    # print("gold_path", len(gold_path))
    # print("gold_moves", len(gold_moves))
    # print("gold", len(gold))
    return enc_sentence, gold_path, gold_moves, gold


# %% [markdown]
# ## Prepare dataset
#

# %%
from datasets import load_dataset

train_dataset = load_dataset("universal_dependencies", "en_lines", split="train")
dev_dataset = load_dataset("universal_dependencies", "en_lines", split="validation")
test_dataset = load_dataset("universal_dependencies", "en_lines", split="test")
print(len(train_dataset))  # type: ignore
print(len(dev_dataset))  # type: ignore
print(len(test_dataset))  # type: ignore


# %%
# Remove non projective trees
train_dataset = [
    sample
    for sample in train_dataset
    if is_projective([-1] + [int(head) for head in sample["head"]])  # type: ignore
]

# %%
# create the embedding dictionary
emb_dictionary = create_dictionary(train_dataset)
print(len(train_dataset))
print(len(dev_dataset))  # type: ignore
print(len(test_dataset))  # type: ignore

# %% [markdown]
# ## DataLoaders
#

# %%
from functools import partial


def prepare_batch(batch_data, get_gold_path=False):
    data = [
        process_sample(s, emb_dictionary, get_gold_path=get_gold_path)
        for s in batch_data
    ]
    # sentences, paths, moves, trees are parallel arrays, each element refers to a sentence
    sentences = [s[0] for s in data]
    paths = [s[1] for s in data]
    moves = [s[2] for s in data]
    trees = [s[3] for s in data]
    # print(f"sentences{len(sentences[0])}")
    # print(f"paths{len(paths[0])}")
    # print("moves", len(moves[0]))
    # print("trees", len(trees[0]))
    return sentences, paths, moves, trees


BATCH_SIZE = 32

train_dataloader = tutils.data.DataLoader(  # type:ignore
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=partial(prepare_batch, get_gold_path=True),
)
dev_dataloader = tutils.data.DataLoader(  # type: ignore
    dev_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=partial(prepare_batch)
)
test_dataloader = tutils.data.DataLoader(  # type:ignore
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=partial(prepare_batch),
)

# %% [markdown]
# ---
#
# # BiLSTM
#

# %%
EMBEDDING_SIZE = 200
LSTM_SIZE = 200
LSTM_LAYERS = 2  # It was 1 before
MLP_SIZE = 200
CLASSES = 4
DROPOUT = 0.2
EPOCHS = 15
LR = 0.002  # learning rate


class Net(nn.Module):
    def __init__(self, device):
        super(Net, self).__init__()
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
                    #print("left")
#------------------------------ firdt condition to check is the left arc -> right arc -> shift -> reduce------------------------------
                    if cond_left:
                        parsers[i].left_arc()
                    else:
                        if cond_right:
                            parsers[i].right_arc()
                        elif cond_reduce:
                            parsers[i].reduce()
                        elif cond_shift:
                            parsers[i].shift()
                        else:
                            print("noMove was possible on left")
#------------------------------ firdt condition to check is the right arc -> shift -> reduce------------------------------
                if moves_argm[i] == 1:
                    #print("right")
                    if cond_right:
                        parsers[i].right_arc()
                    else:
                        if cond_reduce:
                            parsers[i].reduce()
                        elif cond_shift:
                            parsers[i].shift()
                        else:
                            print("noMove was possible on right")
#------------------------------ firdt condition to check is the shift -> reduce------------------------------
                if moves_argm[i] == 2:
                    #print("reduce")
                    if cond_reduce:
                        parsers[i].reduce()
                    elif cond_shift:
                        parsers[i].shift()
                    else:
                        print("noMove was possible on shift")
#------------------------------ firdt condition to check is the reduce and if no reduce was possible take in account the probabilities ------------------------------
                if moves_argm[i] == 3:
                    #print("shift")
                    if cond_shift:
                        parsers[i].shift()
                    else:
                        if moves[i][0] > moves[i][1] and moves[i][0] > moves[i][2] and cond_left:
                            parsers[i].left_arc()
                        else:
                            if moves[i][1] > moves[i][2] and cond_right:
                                parsers[i].right_arc()
                            else:
                                if cond_reduce:
                                    parsers[i].reduce()
                                else:
                                    print(moves[i][0], moves[i][1], moves[i][2], cond_left, cond_right, cond_shift)
                    



# %%model = model.to("xpu")
model = Net(device).to(device)
print(model)


# %%
def evaluate(gold, preds):
    total = 0
    correct = 0

    for g, p in zip(gold, preds):
        for i in range(1, len(g)):
            total += 1
            if g[i] == p[i]:
                correct += 1

    return correct / total


# %%
def train(model, dataloader, criterion, optimizer):
    model.train()  # setup model for training mode
    total_loss = 0
    count = 0
    for batch in dataloader:
        optimizer.zero_grad()
        sentences, paths, moves, trees = batch

        out = model(sentences, paths)
        labels = torch.tensor(sum(moves, [])).to(
            device
        )  # sum(moves, []) flatten the array
        loss = criterion(out, labels)

        count += 1
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    return total_loss / count


def test(model, dataloader):
    model.eval()

    gold = []
    preds = []

    for batch in dataloader:
        sentences, paths, moves, trees = batch
        with torch.no_grad():
            pred = model.infere(sentences)

            gold += trees
            preds += pred

    return evaluate(gold, preds)


# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
model = Net(device).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)


for epoch in range(EPOCHS):
    print("Starting Epoch", epoch)
    avg_train_loss = train(model, train_dataloader, criterion, optimizer)
    val_uas = test(model, dev_dataloader)

    print(
        "Epoch: {:3d} | avg_train_loss: {:5.3f} | dev_uas: {:5.3f} |".format(
            epoch, avg_train_loss, val_uas
        )
    )

# %%
test_uas = test(model, test_dataloader)
print("test_uas: {:5.3f}".format(test_uas))
