import numpy as np
import os
from collections import Counter
import pickle

"""
All the data is already white-space tokenized and lower-cased. one sentence per line.
special words: <bos>, <eos>, <pad>, <unk>
ignore sentences longer than 30
construct a vocabulary consisting of 20K most frequent words in the training set including 4 symbols above
replace out-of-vocabulary words with <unk> before feeding into network
TODO provide the ground truth last word as input to the RNN, not the last word you predicted (???)
"""

def _read_sentences(data_file_path, max_sentence_length):
    """
    Returns:
        list of sentences
    """
    with open(data_file_path, "r", encoding='utf-8') as sentences:
        # TODO: maybe strip special char such as '' or `?
        splitted_sentences = [sentence.strip().split(" ") for sentence in sentences]
        # max_sentence_length includes <bos> and <eos>
        splitted_sentences = [sentence for sentence in splitted_sentences if len(sentence) <= (max_sentence_length-2)]
    return splitted_sentences

def _build_vocab(word_counts, spec_word, max_vocabSize):
    """
    Prune vocabulary to max_vocabSize
    """
    cwd = os.getcwd()
    spec_word_size = len(spec_word)
    words_toKeep = [tupl[0] for tupl in word_counts.most_common(max_vocabSize-spec_word_size)]
    # Create mapping from words/PoS tags to ids
    word_toId = {word: i for i, word in enumerate(words_toKeep, spec_word_size)}
    for i, word in enumerate(spec_word):
        word_toId[word] = i
    id_toWord = {i: word for word, i in word_toId.items()}
    if not os.path.exists(cwd+"/vocab"):
        os.makedirs(cwd+"/vocab")
    with open(cwd+"/vocab/vocab.pkl", "wb") as f:
        pickle.dump(words_toKeep, f)
    with open(cwd+"/vocab/wordIds.pkl", "wb") as f:
        pickle.dump(word_toId, f)
    with open(cwd+"/vocab/idWords.pkl", "wb") as f:
        pickle.dump(id_toWord, f)
    return word_toId

def _word_to_id(sentences, word_toId, max_sentence_length):
    """
    Map words to id in file
    """
    x = [] # list of word ids
    for sentence in sentences:
        word_ids = []
        word_ids.append(word_toId["<bos>"])
        for i in range(max_sentence_length-2):
            if i < len(sentence):
                if sentence[i] in word_toId:
                    word_ids.append(word_toId[sentence[i]])
                else:
                    word_ids.append(word_toId["<unk>"])
            else:
                word_ids.append(word_toId["<pad>"])
        word_ids.append(word_toId["<eos>"])
        x.append(word_ids)
    return np.array(x)
    
def load_data(data_file_path, spec_word=("<bos>", "<eos>", "<pad>", "<unk>"), max_vocabSize=20000, max_sentence_length=30):
    """
    Loads training data, creates vocabulary (including <bos>, <eos>, <pad>, <unk>) 
    and returns the respective ids for words
    """
    # Load data from file
    word_counts = Counter() # Collect word counts
    splitted_sentences = _read_sentences(data_file_path, max_sentence_length)
    for sentence in splitted_sentences:
        # TODO: maybe consider voc for all sentences instead of just training set?
        for word in sentence:
            if word in word_counts:
                word_counts[word] += 1
            else:
                word_counts[word] = 1
    
    # Prune vocabulary to max_vocabSize
    word_toId = _build_vocab(word_counts, spec_word, max_vocabSize)
    
    # Replace each word with id
    x = _word_to_id(splitted_sentences, word_toId, max_sentence_length)
    return x


def load_data_test(data_file_path, spec_word=("<bos>", "<eos>", "<pad>", "<unk>"), max_sentence_length=30):
    """
    Loads test data and vocabulary and returns the respective ids for words
    """
    cwd = os.getcwd()

    # Load vocabulary from training
    if not os.path.exists(cwd+"/vocab"):
        raise FileNotFoundError("You need to run train.py first in order to generate the vocabulary.")
    with open(cwd+"/vocab/wordIds.pkl", "rb") as f:
        word_toId = pickle.load(f)
    splitted_sentences = _read_sentences(data_file_path, max_sentence_length)
    
    # Replace each word with id
    x = _word_to_id(splitted_sentences, word_toId, max_sentence_length)
    return x


def batch_iter(data, batch_size=64, num_epochs=200, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    Args:
        data: [num_sentences, max_sentence_length]
    Return:
        Tuple of arrays, each shaped [batch_size, unrolled_steps]. The second element
        of the tuple is the same data time-shifted to the right by one.

    """
    num_sentences, max_sentence_length = data.shape
    unrolled_steps = max_sentence_length - 1
    num_batches_per_epoch = int((num_sentences-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(num_sentences))
            shuffled_data = data[shuffle_indices, :]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, num_sentences)
            x = shuffled_data[start_index:end_index, 0:unrolled_steps]
            y = shuffled_data[start_index:end_index, 1:unrolled_steps+1]
            yield x, y


if __name__ == "__main__":
    train_data_file_path = os.path.join(os.path.dirname(__file__), "data/sentences.train")
    test_data_file_path = os.path.join(os.path.dirname(__file__), "data/sentences.eval")
#    x = load_data(train_data_file_path)
    y = load_data_test(test_data_file_path)
    batches = batch_iter(y)
    i = 0
    res = 0
    for batch in batches:
        if i == 0:
           res = batch
