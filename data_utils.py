import numpy as np
import os
from collections import Counter
import pickle
import tensorflow as tf

"""
All the data is already white-space tokenized and lower-cased. one sentence per line.
special words: <bos>, <eos>, <pad>, <unk>
ignore sentences longer than 30
construct a vocabulary consisting of 20K most frequent words in the training set including 4 symbols above
replace out-of-vocabulary words with <unk> before feeding into network
TODO provide the ground truth last word as input to the RNN, not the last word you predicted (???)
"""

class Reader():
    def __init__(self, max_vocabSize=20000, max_sentence_length=30):
        self.word_toId = []
        self.id_toWord = []
        self.max_vocabSize = max_vocabSize
        self.max_sentence_length = max_sentence_length
        
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
        cwd = os.getcwd()
        spec_word_size = len(spec_word)
        words_toKeep = [tupl[0] for tupl in word_counts.most_common(self.max_vocabSize-spec_word_size)]
        # Create mapping from words/PoS tags to ids
        self.word_toId = {word: i for i, word in enumerate(words_toKeep, spec_word_size)}
        for i, word in enumerate(spec_word):
            self.word_toId[word] = i
        self.id_toWord = {i: word for word, i in self.word_toId.items()}
        if not os.path.exists(cwd+"/vocab"):
            os.makedirs(cwd+"/vocab")
        with open(cwd+"/vocab/vocab.pkl", "wb") as f:
            pickle.dump(words_toKeep, f)
        with open(cwd+"/vocab/wordIds.pkl", "wb") as f:
            pickle.dump(self.word_toId, f)
        with open(cwd+"/vocab/idWords.pkl", "wb") as f:
            pickle.dump(self.id_toWord, f)
    
    def word_to_id(self, sentences, padding=True, ending=True):
        """
        Map words to id in file
        """
        x = [] # list of word ids
        for sentence in sentences:
            x.append(self.word_toId["<bos>"])
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
            sentence = [self.id_toWord[id] for id in x[i, :] if self.id_toWord[id] not in ["<bos>", "<pad>"]]
            sentences.append(" ".join(sentence))
        return sentences
        
    def raw_data(self, data_path, spec_word=("<bos>", "<eos>", "<pad>", "<unk>")):
        """
        Loads training and evaluation data, creates vocabulary (including <bos>, <eos>, <pad>, <unk>) 
        and returns the respective ids for words
        Args:
            data_file_path: directory that contains data files
        """
        # Load data from file
        train_path = os.path.join(data_path, "sentences.train")
        eval_path = os.path.join(data_path, "sentences.eval")
        word_counts = Counter() # Collect word counts
        train_sentences = self._read_sentences(train_path)
        eval_sentences = self._read_sentences(eval_path)
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
        train_data = self.word_to_id(train_sentences)
        eval_data = self.word_to_id(eval_sentences)
        return train_data, eval_data

    def batch_producer(self, raw_data, batch_size, name=None):
        """
        Generates a batch iterator for a dataset and feed directly to graph
        """
        unrolled_steps = self.max_sentence_length - 1
        with tf.name_scope(name, "BatchProducer", [raw_data, batch_size, unrolled_steps]):
            raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)
    
            data_len = tf.size(raw_data)
            batch_len = data_len // batch_size
            data = tf.reshape(raw_data[0 : batch_size * batch_len], [batch_size, batch_len])
            epoch_size = (batch_len - 1) // unrolled_steps
        
            i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue() # capacity of queue epoch_size
            x = tf.strided_slice(data, [0, i * unrolled_steps], [batch_size, (i + 1) * unrolled_steps]) # like generator
            y = tf.strided_slice(data, [0, i * unrolled_steps + 1], [batch_size, (i + 1) * unrolled_steps + 1])
            x.set_shape([batch_size, unrolled_steps])
            y.set_shape([batch_size, unrolled_steps])
        return x, y

if __name__ == "__main__":
    data_file_path = os.path.join(os.path.dirname(__file__), "data")
    reader = Reader()
#    train_data, eval_data = reader.raw_data(data_file_path)
    x, y = reader.batch_producer(eval_data, 64)