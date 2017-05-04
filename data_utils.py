import numpy as np
import os
from collections import Counter
import pickle
#import tensorflow as tf

"""
All the data is already white-space tokenized and lower-cased. one sentence per line.
special words: <bos>, <eos>, <pad>, <unk>
ignore sentences longer than 30
construct a vocabulary consisting of 20K most frequent words in the training set including 4 symbols above
replace out-of-vocabulary words with <unk> before feeding into network
TODO provide the ground truth last word as input to the RNN, not the last word you predicted (???)
"""

class Reader():
    def __init__(self, data_file_path, max_vocabSize=20000, max_sentence_length=30):
        self.max_vocabSize = max_vocabSize
        self.max_sentence_length = max_sentence_length
        self.data_file_path = data_file_path
        vocab_file = os.path.join(self.data_file_path, "vocab.pkl")
        wordIds_file = os.path.join(self.data_file_path, "wordIds.pkl")
        idWords_file = os.path.join(self.data_file_path, "idWords.pkl")
        if not (os.path.exists(vocab_file) and os.path.exists(wordIds_file) and os.path.exists(idWords_file)):
            print("Preprocessing...")
            self.raw_data()
        else:
            print("Loading preprocessed data...")
            self.load_preprocessed()
        
    def _read_sentences(self, data_file_path):
        """
        Returns:
            list of sentences
        """
        with open(data_file_path, "r", encoding='utf-8') as sentences:
            # TODO: maybe strip special char such as '' or `?
            splitted_sentences = [sentence.strip().split(" ") for sentence in sentences]
            # max_sentence_length includes <bos> and <eos>
            splitted_sentences = [sentence for sentence in splitted_sentences if len(sentence) <= (self.max_sentence_length-2)]
        return splitted_sentences
    
    def _build_vocab(self, word_counts, spec_word):
        """
        Prune vocabulary to max_vocabSize
        """
        spec_word_size = len(spec_word)
        words_toKeep = [tupl[0] for tupl in word_counts.most_common(self.max_vocabSize-spec_word_size)]
        # Create mapping from words/PoS tags to ids
        self.word_toId = {word: i for i, word in enumerate(words_toKeep, spec_word_size)}
        for i, word in enumerate(spec_word):
            self.word_toId[word] = i
        self.id_toWord = {i: word for word, i in self.word_toId.items()}
        if not os.path.exists(self.data_file_path):
            os.makedirs(self.data_file_path)
        with open(os.path.join(self.data_file_path, "vocab.pkl"), "wb") as f:
            pickle.dump(words_toKeep, f)
        with open(os.path.join(self.data_file_path, "wordIds.pkl"), "wb") as f:
            pickle.dump(self.word_toId, f)
        with open(os.path.join(self.data_file_path, "idWords.pkl"), "wb") as f:
            pickle.dump(self.id_toWord, f)
    
    def word_to_id(self, sentences, padding=True, ending=True):
        """
        Map words to id in file
        Return:
            list of lists of id
        """
        x = [] # list of word ids
        for sentence in sentences:
            words = []
            # add <bos>
            words.append(self.word_toId["<bos>"])
            for i in range(self.max_sentence_length-1): # only need to fill self.max_sentence_length words
                if i < len(sentence):
                    if sentence[i] in self.word_toId:
                        words.append(self.word_toId[sentence[i]])
                    else:
                        words.append(self.word_toId["<unk>"])
                elif i == len(sentence):
                    if ending:
                        words. append(self.word_toId["<eos>"])
                else:
                    if padding:
                        words.append(self.word_toId["<pad>"])
            x.append(words)
            """
            for i in range(self.max_sentence_length-2):
                if i < len(sentence):
                    if sentence[i] in self.word_toId:
                        x.append(self.word_toId[sentence[i]])
                    else:
                        x.append(self.word_toId["<unk>"])
                else:
                    if padding:
                        x.append(self.word_toId["<pad>"])
            if ending:
                x.append(self.word_toId["<eos>"])
            """
        return x
    
    def id_to_word(self, x):
        """
        Map id to words
        Args:
            x: [num_sentences, num_words]
        Return:
            sentences: list of strings
        """
        sentences = []
        num_sentences = x.shape[0]
        for i in range(num_sentences):
            sentence = [self.id_toWord[id] for id in x[i, :]]
            sentences.append(" ".join(sentence))
        return sentences
        
    def raw_data(self, spec_word=("<bos>", "<eos>", "<pad>", "<unk>")):
        """
        Loads training and evaluation data, creates vocabulary (including <bos>, <eos>, <pad>, <unk>) 
        and returns the respective ids for words
        Return:
            tuple of list of word ids
        """
        # Load data from file
        train_path = os.path.join(self.data_file_path, "sentences.train")
        eval_path = os.path.join(self.data_file_path, "sentences.eval")
        cont_path = os.path.join(self.data_file_path, "sentences.continuation")
        word_counts = Counter() # Collect word counts
        train_sentences = self._read_sentences(train_path)
        eval_sentences = self._read_sentences(eval_path)
        self.cont_data = self._read_sentences(cont_path)
        for sentence in train_sentences:
            # TODO: maybe consider voc for all sentences instead of just training set?
            for word in sentence:
                if word in word_counts:
                    word_counts[word] += 1
                else:
                    word_counts[word] = 1
        
        # Prune vocabulary to max_vocabSize
        self._build_vocab(word_counts, spec_word)
        
        # Replace each word with id
        self.train_data = self.word_to_id(train_sentences)
        self.eval_data = self.word_to_id(eval_sentences)
        with open(os.path.join(self.data_file_path, "train_data.pkl"), "wb") as f:
            pickle.dump(self.train_data, f)
        with open(os.path.join(self.data_file_path, "eval_data.pkl"), "wb") as f:
            pickle.dump(self.eval_data, f)    
        with open(os.path.join(self.data_file_path, "cont_data.pkl"), "wb") as f:
            pickle.dump(self.cont_data, f)    
    
    def load_preprocessed(self):
        with open(os.path.join(self.data_file_path, "wordIds.pkl"), "rb") as f:
            self.word_toId = pickle.load(f)
        with open(os.path.join(self.data_file_path, "idWords.pkl"), "rb") as f:
            self.id_toWord = pickle.load(f)
        with open(os.path.join(self.data_file_path, "train_data.pkl"), "rb") as f:
            self.train_data = pickle.load(f)
        with open(os.path.join(self.data_file_path, "eval_data.pkl"), "rb") as f:
            self.eval_data = pickle.load(f)
        with open(os.path.join(self.data_file_path, "cont_data.pkl"), "rb") as f:
            self.cont_data = pickle.load(f)
    
    def batch_iter(self, raw_data, batch_size=64, shuffle=True):
        """
        Generates a batch iterator for a dataset.
        Args:
            raw_data: list of lists of ids
        Return:
            Tuple of arrays, each shaped [batch_size, unrolled_steps]. The second element
            of the tuple is the same data time-shifted to the right by one.
    
        """
        data = np.array(raw_data) 
#        data = np.reshape(data, [-1, self.max_sentence_length])
        self.num_batches_per_epoch = num_batches_per_epoch = int(data.shape[0]/batch_size)
        data = data[0:num_batches_per_epoch*batch_size, :]
        num_sentences = data.shape[0]
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(num_sentences))
            shuffled_data = data[shuffle_indices, :]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = (batch_num + 1) * batch_size
            x, y = shuffled_data[start_index:end_index, 0:self.max_sentence_length-1], shuffled_data[start_index:end_index, 1:self.max_sentence_length]
            yield x, y
                
if __name__ == "__main__":
    data_file_path = os.path.join(os.path.dirname(__file__), "data")
    reader = Reader(data_file_path, max_sentence_length=30)
    batches = reader.batch_iter(reader.train_data, 64)
    i = 0
    for batch in batches:
        x, y = batch
        if i == 0:
            break