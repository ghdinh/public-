# Author: Gaby Dinh
# Date: 4/23/2023

import random
import torch
import torch.optim as optim
import tqdm
import argparse 

import preprocess
from model import LSTMCRF, BertModel

### model hyperparameters
EMBED_DIM = 50
# embedding dimension
NUM_HIDDEN = 50
# LSTM hidden dimension
NUM_LAYERS = 2
# number of LSTM layers
BIDIRECTIONAL = True

### training hyperparams
LEARNING_RATE = 0.0002
# number of epochs to train
NUM_EPOCHS = 20
# how many epochs to train before evaluating on dev set
EVAL_EVERY = 20

def evaluate(tagger, eval_dataset, index2tag, word2index):
    """evaluates crf on eval_data
    Args:
      tagger: LSTM-CRF
      eval_dataset: dev_data or test_data
      index2tag: dict[int, str] for converting label ids to original strings
      word2index: dict[int, str] for converting word ids to original strings
    Returns:
      (predicted tag sequence, accuracy)
    """
    num_correct = 0
    total_tags = 0

    preds_list = []
    with torch.no_grad():
        for eval_data in tqdm.tqdm(eval_dataset):
            input_ids = eval_data[0]
            labels = eval_data[1]

            preds = tagger.decode(input_ids)

            num_correct += len([1 for p, l in zip(preds, labels[1:]) if p == l])

            input_ids = input_ids.tolist()
            input_words = [word2index[index] for index in input_ids]

            preds = [index2tag[int(x)] for x in preds]
            for word, tag in zip(input_words, preds):
                preds_list.append(word + " " + tag)
            preds_list.append(" ")

            total_tags += len(preds)


    acc = round(num_correct * 100 / total_tags, 2)
    return preds_list, acc


def evaluate_bert(tagger, eval_dataset, index2tag, word2index):
    """Evaluates bert on eval_data
    Args:
      tagger: Bert
      eval_dataset: dev_data or test_data
      index2tag: dict[int, str] for converting label ids to original strings
      word2index: dict[int, str] for converting word ids to original strings
    Returns:
      (predicted tag sequence, accuracy)
    """
    num_correct = 0
    total_tags = 0

    preds_list = []
    with torch.no_grad():
        for eval_data in tqdm.tqdm(eval_dataset):
            input_ids = eval_data[0]
            labels = eval_data[1]

            loss, logits = tagger(input_ids, labels[1:])
            for i in range(logits.shape[0]):
                logits_sent = logits[i]
                preds = logits_sent.argmax(dim=1)

                num_correct += len([1 for p, l in zip(preds, labels[1:]) if p == l])

                preds = [index2tag[int(x)] for x in preds]

            input_ids = input_ids.tolist()
            input_words = [word2index[index] for index in input_ids]

            for word, tag in zip(input_words, preds):
                preds_list.append(word + " " + tag)
            preds_list.append(" ")

            total_tags += len(preds)

    acc = round(num_correct * 100 / total_tags, 2)
    return preds_list, acc


def train():
    name = 'Final Project Training'
    print(name)

    # 2. data loading
    train_data, train_labels = preprocess.process_file('train_new.txt')
    dev_data, dev_labels = preprocess.process_file('dev_new.txt')
    test_data, test_labels = preprocess.process_file('test_new.txt')

    # 3. vocabs
    all_vocabs, tag2index, index2tag, word2index = preprocess.collect_vocabs(train_data + dev_data + test_data)

    # 4. tensorized data
    train_dataset = preprocess.vectorize(train_data, train_labels, all_vocabs, tag2index)
    dev_dataset = preprocess.vectorize(dev_data, dev_labels, all_vocabs, tag2index)
    test_dataset = preprocess.vectorize(test_data, test_labels, all_vocabs, tag2index)

    # 5. model init
    tagger = LSTMCRF(
        source_vocabs=all_vocabs,
        tag_to_index=tag2index,
        embed_dim=EMBED_DIM,
        hidden_dim=NUM_HIDDEN,
        num_layers=NUM_LAYERS,
        bidirectional=BIDIRECTIONAL
    )

    tagger0 = BertModel(
        tag2index=tag2index
    )

    # 6. optimizer init
    optimizer = optim.Adam(tagger.parameters(), lr=LEARNING_RATE)
#    optimizer = optim.Adam(tagger0.parameters(), lr=LEARNING_RATE)

    # 7. training loop
    print("Begin training")

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_loss = 0.

        random.shuffle(train_dataset)
        for input_ids, labels in tqdm.tqdm(train_dataset, desc=f'[Epoch {epoch}/{NUM_EPOCHS}]'):

            optimizer.zero_grad()
            loss = tagger.nll_loss(input_ids, labels)
#            loss, logits = tagger0(input_ids, labels[1:])

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # info at end of epoch
        log = f'[Epoch {epoch}/{NUM_EPOCHS}] Loss: {epoch_loss:.2f}'
        preds_count = 0
        if epoch % EVAL_EVERY == 0:
            print("Begin evaluation on dev set")
            dev_preds_list, dev_acc = evaluate(tagger, dev_dataset, index2tag, word2index)
            test_preds_list, test_acc = evaluate(tagger, test_dataset, index2tag, word2index)
#            preds_list, dev_acc = evaluate_bert(tagger0, dev_dataset, index2tag, word2index)
            log = f'{log} | Dev Acc: {dev_acc}%'
            log = f'{log} | Test Acc: {test_acc}%'
            preprocess.preds_to_file(dev_preds_list, preds_count)
            preds_count += 1
            preprocess.preds_to_file(test_preds_list, preds_count)
        print(log)


if __name__ == '__main__':
    train()
