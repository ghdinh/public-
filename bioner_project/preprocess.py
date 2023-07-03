# Author: Gaby Dinh
# Date: 04/23/23 18:16

"""helper functions"""
import torch
import tqdm

### special vocabs
UNK = '<unk>'  # `unknown`, for tokens not in the train vocab
BOS = '<bos>'  # `beginning of sequence`, for use in CRF
EOS = '<eos>'  # `end of sequence`, for use in CRF

NER_list = ["I-ety", "B-ety", "O"]


def process_file(file_path):
    total_text = []
    total_NER = []
    with open(file_path, encoding='utf8') as f:
        words = []
        labels = []
        for line in f:
            if line == "" or line == "\n":
                total_text.append(words)
                total_NER.append(labels)
                words = []
                labels = []
            else:
                splits = line.split()
                words.append(splits[0])
                if splits[-1] in NER_list:
                    labels.append(splits[-1])
    return total_text, total_NER


def collect_vocabs(training_data):
    """collects source (sentence) and target (POS tags) vocabs

    `tgt_vocabs_inv` is the inverse of `tgt_vocabs` and is useful for converting
    model predictions (which are originally ints) back to strings
    """
    print("Collecting vocabs")
    vocabs_list = set()
    for sent in tqdm.tqdm(training_data):
        for word in sent:
            vocabs_list.add(word)

    sorted_vocabs_list = sorted(vocabs_list)

    # word2index
    vocab_dict = {x: i for i, x in enumerate([UNK] + sorted_vocabs_list)}

    # index2vocab
    index2word = {}
    for key, value in vocab_dict.items():
        index2word[value] = key

    # tag2index, index2tag
    tag2index = {BOS: 0, 'B-ety': 1, 'I-ety': 2, 'O': 3, EOS: 4}
    index2tag = {0: BOS, 1: 'B-ety', 2: 'I-ety', 3: 'O', 4: EOS}

    return vocab_dict, tag2index, index2tag, index2word


def vectorize(sents, labels, word2index, tag2index):
    """converts strings into vectors using word2index and tag2index

    Args:
        sents
        labels
        word2index
        tag2index

    Returns:
      tensorized data
    """

    out = []
    for token, tag in zip(sents, labels):
        curr_sent = []
        # tensorized BIO tags, prefixed with <bos>
        tags = [tag2index[x] for x in [BOS] + tag]

        tag_tensor = torch.tensor([tags], dtype=torch.int64)
        curr_sent.append(tag_tensor)

        # tensorized source data (sentence tokens)
        words = [word2index.get(x, word2index[UNK]) for x in token]
        # (1, src_len)
        words_tensor = torch.tensor([words], dtype=torch.int64)

        curr_sent.insert(0, words_tensor)
        out.append(curr_sent)

    return out


def process_sent_tag(text, labels):
    data = []
    for sent, tags in zip(text, labels):
        data.append((sent, tags))
    return data

def preds_to_file(preds_list, preds_count):
    output_text = ""

    with open(f"preds{preds_count}.txt", 'w', encoding="utf-8") as fp:
        for item in preds_list:
            # write each item on a new line
            #    fp.write(str(item) + "\n")
            fp.write(f"{item}\n")
        print('Done')
    # convert to file
    return output_text
