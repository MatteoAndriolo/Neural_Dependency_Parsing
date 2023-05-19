# %% [markdown]
# ### The "Auto" Classes of Hugginface
#
# *  The first thing that we need to do is to download a pretrained model and its tokenizer
# * We need to use the classes
#   * ```AutoModel```
#   *  ```AutoTokenizer```
# * We can load both using the ```from_pretrained``` function
# * The opposite function is the ```save_pretrained``` function, which can be used to save a model to disk
#
#
#

# %%
from transformers import AutoTokenizer, AutoModel

## Encoder-only models
model_name = "bert-base-uncased"

model = AutoModel.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# %% [markdown]
# Tokenizers outputs
# The tokenizer applies the following steps:
#   1. Preprocesses the text and tokenizes it in subwords
#   2. Associates to every subword an ```input_id``` which is used to fetch its embedding in the embedding layer (or layer 0)
#   3. Adds ```attention_mask``` and ```token_type_ids```
string = "I love tokenization"
print(string)
print(tokenizer(string))

# After tokenizing into subwords, each subword is associated to a ```input_id``` which tells the model which embedding to get for that subword
# You can think of it as a row index in the embedding matrix
#  * love ➡️ 2293 ➡️ get embedding in row 2293

print(tokenizer.tokenize(string, add_special_tokens=True))

output = tokenizer(string)
print(tokenizer.convert_ids_to_tokens(output["input_ids"]))
# ['[CLS]', 'i', 'love', 'token', '##ization', '[SEP]']

output = tokenizer(string)
print(tokenizer.decode(output["input_ids"]))
# '[CLS] i love tokenization [SEP]'


# %% The ATTENTION MASK
# Suppose that we have 2 sentences of different lengths
# If the sentences are in the same batch, the shortest one needs to be padded: we need to append [PAD] tokens to the shortest sentence so that they have the same length
# * However, we don't want the self-attention to operate over the [PAD] tokens
# * tokenizers handles all of this for us
#
#

sentences = ["I love tokenization", "I really like the city of Padua"]
output = tokenizer(sentences, padding=True, return_tensors="pt")

print(output["input_ids"])
print(output["attention_mask"])

print(tokenizer.convert_ids_to_tokens(output["input_ids"][0]))
print(tokenizer.convert_ids_to_tokens(output["input_ids"][1]))

# %% [markdown]
# The ```token_type_ids```
#
# Recall that the input embeddings to a transformers are the result of a sum of three elements:
# * Token embeddings: the embeddings that are extracted from the embedding matrix using ```input_ids```
# * Positional embeddings: this are sinusoidal or learned and give the transformer the position information
# * Segment embeddings: when we are doing sentence-pair tasks, i.e. the input consists of $ [CLS] \; seq_A \; [SEP] \; seq_B \; [SEP]$, we may want to add to each embedding the information on the originating sentence

output = tokenizer(
    "The sun is shining today", "Today it's rainy"
)  # sentence-pair tokenization
print(output)

print(tokenizer.decode(output["input_ids"]))

output_single_sent = tokenizer(
    ["The sun is shining today", "Today it's rainy"]
)  # single-sentence tasks
output_sent_pair = tokenizer(
    "The sun is shining today", "Today it's rainy"
)  # sentence-pair tasks

print(tokenizer.batch_decode(output_single_sent["input_ids"]))
print(tokenizer.decode(output_sent_pair["input_ids"]))

# %% [markdown]
# ### The transformer model in Huggingface

# model = AutoModel.from_pretrained("bert-base-uncased")

model

# %%
model.encoder.layer[5]


# %% ### Feeding a batch to a transformer
sequences = [
    "Using transformers is quite simple",
    "Natural Language Processing is the coolest area of AI",
    "BERT is an encoder-only model",
]
batch = tokenizer(sequences, padding=True, return_tensors="pt")
print("input_ids ", batch["input_ids"], "\n\n")
print("attention_mask ", batch["attention_mask"], "\n\n")
print("token_type_ids ", batch["token_type_ids"], "\n\n")

tokenizer.batch_decode(batch["input_ids"])

model_output = model(
    **batch, output_hidden_states=True
)  # equivalent to model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], token_type_ids=batch['token_type_ids'])
all_states = (
    model_output.hidden_states
)  # list of outputs from all transformer layers, layer 0, 1, 2, ...., 12 (layer 0 is the embedding layer)
print(len(all_states))
print(all_states[3])  # output of 4th layer

print(model_output.last_hidden_state)
print(
    "batch_size x seq_len x hidden_dim", model_output.last_hidden_state.shape, sep="\n"
)

# %% [markdown]
# ## Datasets

# %% [markdown]
# ### Loading a dataset from the Huggingface Hub
#
# - We are using the MRPC (Microsoft Research Paraphrasing Corpus) dataset that is part of the General Language Understanding Evaluation (GLUE) benchmark
# - Task: given two sentences, assign positive class (1) if the two sentences are paraphrases of one another (assign 0 otherwise)
# - [Link](https://huggingface.co/datasets/glue) to the dataset on the hub

# %%
from datasets import load_dataset

mrpc_dataset = load_dataset("glue", "mrpc")
mrpc_dataset

# %% # Let's check the dataset features and examples

print(mrpc_dataset["train"].features)

print(mrpc_dataset["train"][0], end="\n\n")
print(mrpc_dataset["train"][1])

# %% [markdown]
# ### Operations on datasets

# %% [markdown]
# The datasets library supports various operations on datasets such as slicing and column selection

# %%
mrpc_dataset["train"][:10]  # Slicing

mrpc_dataset["train"]["label"]  # Column selection

# ```select``` returns rows according to a list of indices:
sel_data = mrpc_dataset["train"].select([0, 11, 22, 1514])

# ```filter``` returns rows that match a specified condition:

filtered_data = mrpc_dataset["train"].filter(
    lambda example: example["sentence1"].startswith("This")
)
filtered_data[:-1]

# %% Train and test splitting

mrpc_dataset["train"].train_test_split(test_size=0.1)

# %% [markdown]
# The map function is one of the most important functions: it applies a function to every example in the dataset. It's primary usage in NLP is for tokenization

# %%
new_dataset = mrpc_dataset["train"].map(
    lambda example: {"new_sentence1": "Sentence 1: " + example["sentence1"]}
)  # adds a new column as a function of the previous one, inside each item
new_dataset[0]

# %% [markdown]
# ### Saving and loading datasets from disk

from datasets import load_from_disk

save_path = "datasets/my_data"

new_dataset.save_to_disk(save_path)  # Saving
reloaded_dataset = load_from_disk(save_path)  # Loading

# %% [markdown]
# ### Other functions

# %% [markdown]
# Other useful functions:
#
# - [Rename, remove, cast, and flatten](https://huggingface.co/docs/datasets/v2.12.0/en/process#rename-remove-cast-and-flatten)
# - [Concatenate, interleave](https://huggingface.co/docs/datasets/v2.12.0/en/process#concatenate)

# %% [markdown]
# Datasets can also be exported to:
#   - csv
#   - json
#   - parquet
#   - sql
#   - In-memory objects Pandas Dataframe / Python Dictionary
#
# See [Guide to export](https://huggingface.co/docs/datasets/v2.12.0/en/process#export)

# %% [markdown]
# ## Fine-tuning a transformer model

# %% [markdown]
# Fine-tuning transformer models:
#
#   - We want to fine-tune BERT on the MRPC (Microsoft Research Paraphrase Corpus) dataset
#   - The first thing to do is to preprocess the data using tokenizer:
#     - The task is a **sentence-pair classification task**
#     - We need every input to be in the format
# $ [CLS] \; seq_1 \; [SEP] \; seq_2 \; [SEP]$
#
#     - We will feed the embedding of the [CLS] to the classification head on top of BERT

# %% [markdown]
# ### Dataset tokenization


# %%
# This is the function that we want to apply on the entire dataset
def tokenize(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


# # We can apply on a single split (train/val/test)
# train_dataset = mrpc_dataset["train"].map(tokenize, batched=True)

# We can also apply the tokenization on the DatasetDict, so we will tokenize train, val and test with a single line
mrpc_dataset = mrpc_dataset.map(tokenize, batched=True)

# %% [markdown]
# ### Dynamic padding and collators
# Before training, there are two things we need to take care of.
#
# * dynamic padding: pad to the longest sentence in the **batch**
# * Conversion to PyTorch tensors

from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")


samples = mrpc_dataset["train"][:8]
samples = {
    "input_ids": samples["input_ids"],
    "attention_mask": samples["attention_mask"],
    "token_type_ids": samples["token_type_ids"],
    "label": samples["label"],
}
data_collator(samples)

# %% [markdown]
# ### Different models for different tasks

# %% [markdown]
# - We need to define the model that we want to train
# - Before, we instatiated a ```AutoModel``` object
#   - This one is unsuitable for classification tasks:
#     -it only contains the BERT encoder but no classification head
# - The ""classification head"" is a feed-forward neural network placed on top of BERT to solve the classification task
# - For our task we will use ```AutoModelForSequenceClassification```
# - Other tasks require different models: ```AutoModelForTokenClassification```, ```AutoModelForQuestionAnswering```

# %%
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=2
)
print(model)

# %% [markdown]
# For training, we will be using the Trainer API from Huggingface.
#
# -  It will handle all details of training and validation
# -  The [```Trainer```](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Trainer) object requires a [```TrainingArguments```](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments) object that contains all the specifications of the training procedure
#
#

# %%
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="bert_mrpc",  # where to save checkpoints and model predictions
    per_device_train_batch_size=8,  # training batch size
    per_device_eval_batch_size=16,  # validation/test batch sizie
    num_train_epochs=3,  # number of epochs
    save_strategy="epoch",  # checkpoint saving frequency
    evaluation_strategy="epoch",  # how frequently to run validation, can be epoch or steps (in this case you need to specify eval_steps)
    metric_for_best_model="f1",  # metric used to pick checkpoints
    greater_is_better=True,  # whether the metric for checkpoint needs to be maximized or minimized
    learning_rate=3e-5,  # learning rate or peak learning rate if scheduler is used
    optim="adamw_torch",  # which optimizer to use
    lr_scheduler_type="linear",  # which scheduler to use
    warmup_ratio=0.1,  # % of steps for which to do warmup
    seed=33,  # setting seed for reproducibility
    load_best_model_at_end=True,
)  # after training, load the best checkpoint according to metric_for_best_model


# %% [markdown]
#
# - Other parameters to consider for your projects:
#   - ```gradient_accumulation_steps``` and ```gradient_checkpointing``` if you have memory problems (batch doesn't fit in memory)
#       - See [here](https://huggingface.co/docs/transformers/v4.18.0/en/performance#gradient-accumulation) for a full explanation of what they do
#   - ```report_to```  if you want to report your metrics, train loss to an external logger such as Weights & Biases (highly reccomended for your projects, see [wandb.ai](https://wandb.ai))
#   - ```resume_from_checkpoint```: you can restart training from a checkpoint by passing the save path here
#
#
#

# %% [markdown]
# Let's define an evaluation function to be used during validation (and later for test)

# %%
import evaluate
import numpy as np


def compute_metrics_mrpc(eval_preds):
    metric = evaluate.load("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


# %%
from transformers import Trainer

trainer = Trainer(
    model,
    training_args,
    train_dataset=mrpc_dataset["train"],
    eval_dataset=mrpc_dataset["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics_mrpc,
)

# %%
trainer.train()

# %% [markdown]
# ### Computing performance on the test set

# %%
test_predictions = trainer.predict(mrpc_dataset["test"])
print(test_predictions.metrics)

# %%
trainer.state.best_model_checkpoint  # folder where best model is saved

# %% [markdown]
# ## Further reading/learning
#
# - In this tutorial, we have covered parts of chapter 1-3 of the HuggingFace Course
#     - Check the entire course [here](https://huggingface.co/learn/nlp-course/chapter1/1) to have a full understading of the library
#     - Particularly, check how you can solve other tasks with HuggingFace transformers (NER, QA, Summarization) [here](https://huggingface.co/learn/nlp-course/chapter7/1?fw=pt)
# - You should definitely learn to use a logger with your experiments
#     - A logger tracks multiple metrics (loss, accuracy, F1) for every run and gives you the ability to compare different experiments
#     - A good example of this is Weights & Biases (WandB): check the getting started [here](https://docs.wandb.ai/guide)
