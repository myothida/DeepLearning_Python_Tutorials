import torch
import numpy as np

class CharTokenizer:
  def __init__(self, vocabulary):
    self.token_id_for_char = {char: token_id for token_id, char in enumerate(vocabulary)}
    self.char_for_token_id = {token_id: char for token_id, char in enumerate(vocabulary)}

  @staticmethod
  def train_from_text(text):
    vocabulary = set(text) # remove duplicates
    return CharTokenizer(sorted(list(vocabulary)))

  def encode(self, text):
    token_ids = []
    for char in text:
      token_ids.append(self.token_id_for_char[char])
    return torch.tensor(token_ids, dtype=torch.long)

  def decode(self, token_ids):
    chars = []
    for token_id in token_ids.tolist():
      chars.append(self.char_for_token_id[token_id])
    return ''.join(chars)


  def vocabulary_size(self):
    return len(self.token_id_for_char)


class WordTokenizer:
    def __init__(self, vocabulary):
        self.token_id_for_word = {word: token_id for token_id, word in enumerate(vocabulary)}
        self.word_for_token_id = {token_id: word for token_id, word in enumerate(vocabulary)}

    @staticmethod
    def train_from_text(text):
        """Create a tokenizer from the unique words in the given text."""
        words = set(text.split())  # Split text into words and remove duplicates
        return WordTokenizer(sorted(list(words)))

    def encode(self, text):
        """Convert a string of text into a tensor of token IDs."""
        token_ids = [self.token_id_for_word[word] for word in text.split()]
        return torch.tensor(token_ids, dtype=torch.long)

    def decode(self, token_ids):
        """Convert a tensor of token IDs back into a string of text."""
        return ' '.join([self.word_for_token_id[token_id] for token_id in token_ids.tolist()])

    def vocabulary_size(self):
        """Return the size of the tokenizer's vocabulary."""
        return len(self.token_id_for_word)


class NGramsTokenizer:
    def __init__(self, n):
        self.n = n

    def generate_ngrams(self, tokens):
        """Generate n-grams from a sequence of tokens."""
        ngrams = [
            tokens[i:i+self.n]
            for i in range(len(tokens) - self.n + 1)
        ]
        return ngrams
