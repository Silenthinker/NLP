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
        with tf.variable_scope("model", reuse=None):
            self.model = Model(self.config, training=False)
        self.args = args
        
    def sample(self, sess, beg=['i']):
        return self.model.sample(sess, self.id_toWords, self.word_toId, self.args.n, beg)
        
def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n', type=int, default=20, help='max generated sentence length')
    parser.add_argument('-beg', type=str, default="", help='beginning of a sentence')
    parser.add_argument('--save_dir', type=str, default='save', help='model directory to store checkpointed models')
    parser.add_argument('--data_dir', type=str, default='data/', help='data directory containing training, evaluation, and continuation data')
    parser.add_argument('--res_dir', type=str, default='res', help='data directory containing results')
    
    args = parser.parse_args()
    with open(os.path.join(args.data_dir, 'cont_data.pkl'), 'rb') as f:
        cont_data = pickle.load(f)
    if not os.path.isdir(args.res_dir):
        os.makedirs(args.res_dir)
    res_file = os.path.join(args.res_dir, "group10.continuation")
    f = open(res_file, "w")
    with tf.Graph().as_default():
        sampler = Sampler(args)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(tf.global_variables())
            ckpt = tf.train.get_checkpoint_state(args.save_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                if args.beg:
                    print(sampler.sample(sess, beg=args.beg.split(' ')))
                else:
                    for idx, sentence in enumerate(cont_data):
                        print("Generating {}/{} sentence".format(idx, len(cont_data)))
                        f.write(sampler.sample(sess, beg=sentence))
                        f.write("\n")
    f.close()
                
if __name__ == '__main__':
    main()