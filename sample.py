from __future__ import print_function
import tensorflow as tf
import argparse
import os
import pickle
from model import Model

class Sampler():
    def __init__(self, args):
        # load data
        with open(os.path.join(args.save_dir, 'config.pkl'), 'rb') as f:
            self.config = pickle.load(f)
        with open(os.path.join(args.data_dir, 'wordIds.pkl'), 'rb') as f:
            self.word_toId = pickle.load(f)
        with open(os.path.join(args.data_dir, 'idWords.pkl'), 'rb') as f:
            self.id_toWords = pickle.load(f)
        self.model = Model(self.config, training=False)
        self.args = args
        
    def sample(self, sess, beg=['i']):
        return self.model.sample(sess, self.id_toWords, self.word_toId, self.args.n, beg)
        
def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--save_dir', type=str, default='save', help='model directory to store checkpointed models')
    parser.add_argument('-n', type=int, default=20, help='max generated sentence length')
    parser.add_argument('--data_dir', type=str, default='data/', help='data directory containing training, evaluation, and continuation data')
    args = parser.parse_args()
    
    sampler = Sampler(args)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(args.save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print(sampler.sample(sess))

if __name__ == '__main__':
    main()
