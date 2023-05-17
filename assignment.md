# Project Assignment
## Introduction
This year the second part of the project will be on **neural transition-based parsing for dependency grammars**. You need to look into the `arc-eager parsing algorithm`, which was briefly mentioned in lecture 10, and the static oracle traditionally used for this parser. All you need to know about this can be found in section 2 of the article at this link.

## Assignment
You are required to
* create a baseline model which uses `biLSTM` for extracting features from the input words, _in a way very similar to what has been done for the arc-standard parser in the second open lab session_. 
* develop a second model using `BERT` in place of the biLSTM. Be aware that BERT assigns embeddings to each token derived using `BPE`: 
  * in order to derive embeddings for words that are split into several tokens, you can average the embeddings of the tokens, or else just take the embedding of the left-most token for the word.

Use any library of your choice for the **implementation and the training** of biLSTM and for the implementation and the fine-tuning of BERT. For instance you can use the Hugging Face library, which will be presented in the third open lab session.

For the **evaluation**, use a dependency treebank from the Universal Dependency (UD) project. There is no restriction on the choice of the language, but consider also point 6. below, to make sure that your choice can be compared with some SotA result.

As for computational resources, you should be able to do fine-tuning for BERT using Google Colab, given the average dimension of available UD treebanks. However, if you have any problem with this and you have an account at DEI, you can use the Blade Computing Cluster. Description of these computational resources can be found at this link (English instructions in the second part of the document).

> The second part of the project should be presented as a notebook which should be uploaded in the Google shared folder, see the 'Project registration' post in this forum. The notebook should contain the following sections.

## Final structure required
1.   Dataset analysis: describe the treebank and report any useful information, as for instance sentence length distribution.
2.   Description of baseline model and BERT-based model.
3.   Data set-up and training.
4.   Evaluation: comparison between the two models and some error analysis for the BERT-based model.
5.   Brief discussion of SotA for this task: search the leaderboard for the dataset of your choice.
