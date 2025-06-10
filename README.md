# ner-lstm-transformer

Named Entity Recognition (NER) using LSTM and Transformer Models

This project implements and compares two neural network architectures (BiLSTM and Transformer) for Named Entity Recognition (NER), as part of COMP534 Assessment 3 at the University of Liverpool.

⸻

## Dataset

The dataset contains approximately 1,696 sentences, where each word is labeled with one of five entity tags:
	•	I-LOC: Location
	•	I-PER: Person
	•	I-ORG: Organization
	•	I-MISC: Miscellaneous
	•	O: Outside

Each line in the dataset consists of a word and its corresponding tag, separated by a space. Sentences are separated by blank lines.

⸻

## Experimental Protocol
	•	The dataset is split into:
	•	Training set: 70%
	•	Validation set: 15%
	•	Test set: 15%
	•	Padding (<PAD>) and unknown (<UNK>) tokens are added to handle sequence length and OOV words.
	•	Maximum sentence length is based on the longest sentence in the training set.

⸻

## Preprocessing
	•	Built vocabularies (word2idx, tag2idx, idx2tag) with special tokens.
	•	Encoded words and tags into indices.
	•	Padded sequences to the maximum sentence length.
	•	Created custom PyTorch Dataset and DataLoader classes for batching.

⸻

## Models

1. BiLSTM Model
	•	Word embedding layer
	•	Bidirectional LSTM
	•	Fully connected linear layer for tag classification

2. Transformer Model
	•	Word embedding layer with positional encoding
	•	Stack of Transformer encoder layers
	•	Fully connected layer for classification

Shared Settings
	•	Embedding dimension: 100
	•	LSTM hidden size: 128
	•	Transformer: 2 layers, 4 attention heads, 512 hidden dimension
	•	Loss function: CrossEntropyLoss (ignoring <PAD> tokens)
	•	Optimizer: Adam
	•	Epochs: 20
	•	Batch size: 32

⸻

## Training Results

## LSTM Model

Epoch	  Train Acc 	Val Acc	  Train Loss   Val Loss
1	       0.7855	    0.8255	    0.9666	    0.6813
…	          …	         …	         …	         …
20	     0.9990	    0.8688	    0.0135	    0.6144

## Transformer Model

Epoch	  Train Acc	  Val Acc	  Train Loss   Val Loss
1	       0.8140	    0.8255	   0.7117	      0.6955
…	          …	         …	        …	           …
20	     0.9952	    0.8699	   0.0186	      0.7781

The LSTM model showed better generalization and lower validation loss.

⸻

## Final Evaluation (Test Set)

Model evaluated: BiLSTM

Metric	Score
Accuracy	0.8720
Precision	0.8560
Recall	0.8720
F1 Score	0.8617
Test Loss	0.6046

⸻

## Conclusion
	•	Best Model: BiLSTM
	•	Why? Lower test loss and better performance metrics than Transformer on this dataset.

⸻

## Tech Stack
	•	Python
	•	PyTorch
	•	scikit-learn
	•	Matplotlib / Seaborn
