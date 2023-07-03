# Author: Gaby Dinh
# Date: 4/24/2023
import torch
import torch.nn as nn
from preprocess import BOS, EOS

from transformers import BertForTokenClassification


class LSTMEncoder(nn.Module):
    """LSTM model"""

    def __init__(self, num_vocabs, embed_dim, hidden_dim, num_tags, num_layers,
                 bidirectional=False):
        """
        Args:
          num_vocabs: number of all vocabs
          embed_dim: embedding dimension
          hidden_dim: LSTM hidden dimension
          num_tags: number of BIO tags
          num_layers: number of LSTM layers
          bidirectional: whether to make LSTM bidirectional
        """
        super().__init__()

        self.embedding = nn.Embedding(num_vocabs, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True,
                            bidirectional=bidirectional)
        if bidirectional is True:
            self.linear = nn.Linear(hidden_dim * 2, num_tags)
        else:
            self.linear = nn.Linear(hidden_dim, num_tags)

    def forward(self, input_ids):
        """
        Args:
          input_ids: sentence input vector, dimension [1, sent_length]
        Returns:
          emission features, dimension [sent_length, num_tags]
        """
        input_embeds = self.embedding(input_ids)

        lstm_hidden, _ = self.lstm(input_embeds)

        emission = self.linear(lstm_hidden).squeeze(0)
        return emission

    def decode(self, input_ids):
        """
        Args:
          input_ids: sentence input vector [1, sent_length]
        Returns:
          decoded tag sequence
        """
        # dim: [sent_length, num_tags]
        emission = self(input_ids)

        output_list = []

        for step in range(len(emission)):
            best_tag = torch.argmax(emission[step])
            output_list.append(best_tag)
        return output_list

    def nll_loss(self, input_ids, labels):
        """
        Args:
          input_ids: sentence input, [1, sent_len]
          labels: BIO tags, [1, sent_len+1]

        Returns:
          negative log likelihood
        """
        # [seq_length, num_tags]
        emission = self(input_ids) 

        # the negative loglikelihood loss of a sentence is the sum of the loss for individual word tokens in the
        # sentence
        criterion = nn.CrossEntropyLoss()
        labels = labels[0, 1:]
        loss = criterion(emission, labels) 
        return torch.sum(loss)


class LSTMCRF(nn.Module):
    """
    LSTM-CRF sequential tagger (with Viterbi decoding)
    """

    def __init__(self, source_vocabs, tag_to_index, embed_dim, hidden_dim, num_layers,
                 bidirectional=True):
        super().__init__()

        self.num_vocabs = source_vocabs
        self.tag_to_index = tag_to_index
        self.num_tags = len(tag_to_index)
        self.hidden_dim = hidden_dim

        self.lstm = LSTMEncoder(
            num_vocabs=len(source_vocabs),
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_tags=len(tag_to_index),
            num_layers=num_layers,
            bidirectional=bidirectional
        )
        self.transitions = nn.Parameter(torch.rand(self.num_tags, self.num_tags))
        self.transitions.data[tag_to_index[BOS], :] = -1000.
        self.transitions.data[:, tag_to_index[EOS]] = -1000.

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))

    def forward(self, emission):
        """
        Args:
          emission: output of LSTM [sent_length, num_tags]

        Returns:
          alpha, the forward variable
        """

        # forward computes alpha, the partition
        init_alphas = torch.full([self.num_tags], -10000.)

        # beginning of sentence
        init_alphas[self.tag_to_index[BOS]] = 0.
        forward = [init_alphas]

        for index in range(emission.shape[0]):
            emission_score = torch.stack([forward[index]] * emission.shape[1])
            score_to_transition = torch.unsqueeze(emission[index], 0).transpose(0, 1)
            next_tag = emission_score + score_to_transition + self.transitions
            forward.append(torch.logsumexp(next_tag, dim=1))
        terminal_var = forward[-1] + self.transitions[self.tag_to_index[EOS]]
        terminal_var = torch.unsqueeze(terminal_var, 0)
        alpha = torch.logsumexp(terminal_var, dim=1)[0]
        return alpha

    def decode(self, input_ids):
        """
        Args:
          input_ids: sentence input vector [1, sent_length]

        Returns:
          decoded tag sequence
        """
        # emission = (src_len, num_tags)
        emission = self.lstm(input_ids)
        scores = torch.zeros(emission.shape[1])
        backpointers = torch.zeros(emission.shape)
        backpointers = backpointers.type(torch.int)
        scores = scores + emission[0]
        # get scores and paths for each step in sequence
        for i in range(1, emission.shape[0]):
            transition_scores = scores.unsqueeze(1).expand_as(self.transitions) + self.transitions
            max_scores, backpointers[i] = torch.max(transition_scores, 0)
            scores = emission[i] + max_scores
        # get best path
        best_path = [scores.argmax()]
        for path in reversed(backpointers[1:]):
            best_path.append(path[best_path[-1]])
        best_path.reverse()
        return best_path

    def score(self, emission, labels):
        """
        Args:
          emission: output of LSTM, [sent_len, num_tags]
          labels: gold POS tags, [1, sent_len+1]

        Returns:
          score
        """
        score = torch.zeros(1)
        for i, emission in enumerate(emission):
            score = score + self.transitions[labels[i + 1], labels[i]] + emission[labels[i + 1]]
        score = score + self.transitions[self.tag_to_index[EOS], labels[-1]]
        return score

    def nll_loss(self, input_ids, labels):
        """
        Args:
          input_ids: sentence input ids, [1, sent_len]
          labels: BIO tag (target) ids, [1, sent_len+1]

        Returns:
          negative log likelihood
        """
        # [sent_len, num_tags]
        emission = self.lstm(input_ids)
        forward_score = self.forward(emission)
        gold_score = self.score(emission, labels)
        return forward_score - gold_score


class BertModel(torch.nn.Module):

    def __init__(self, tag2index):
        super(BertModel, self).__init__()

        self.bert = BertForTokenClassification.from_pretrained('bert-base-cased', num_labels=len(tag2index))

    def forward(self, vocabs, label):

        output = self.bert(input_ids=vocabs, labels=label, return_dict=False)

        return output
