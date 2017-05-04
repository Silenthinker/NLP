from __future__ import print_function
import tensorflow as tf
import argparse
import time
import os
import pickle

from data_utils import Reader
from model import Model



def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # model
    parser.add_argument('--vocab_size', type=int, default=20000, help='size of vocabulary')
    parser.add_argument('--hidden_size', type=int, default=512, help='size of RNN hidden state')
    parser.add_argument('--embedding_size', type=int, default=100, help='size of embedding')
    parser.add_argument('--unrolled_steps', type=int, default=29, help='RNN unrolled length')
    parser.add_argument('--grad_clip', type=float, default=10., help='clip gradients at this value')
    parser.add_argument('--learning_rate', type=float, default=0.002, help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.97, help='decay rate for rmsprop')
    # training
    parser.add_argument('--batch_size', type=int, default=64, help='minibatch size')
    parser.add_argument('--num_epochs', type=int, default=3, help='number of epochs')
    parser.add_argument('--inter_threads', type=int, default=4, help='inter_op_parallelism_threads')
    parser.add_argument('--intra_threads', type=int, default=4, help='intra_op_parallelism_threads')
    # save and restore
    parser.add_argument('--data_dir', type=str, default='data/', help='data directory containing training, evaluation, and continuation data')
    parser.add_argument('--save_dir', type=str, default='save', help='directory to store checkpointed models')
    parser.add_argument('--log_dir', type=str, default='logs', help='directory to store tensorboard logs')
    parser.add_argument('--save_every', type=int, default=500,
                        help='save frequency')
    parser.add_argument('--init_from', type=str, default=None,
                        help="""continue training from saved model at this path. Path must contain files saved by previous training process:
                            'config.pkl'        : configuration;
                            'chars_vocab.pkl'   : vocabulary definitions;
                            'checkpoint'        : paths to model file(s) (created by tf).
                                                  Note: this file contains absolute paths, be careful when moving files around;
                            'model.ckpt-*'      : file(s) with model definition (created by tf)
                        """)
    args = parser.parse_args()
    train(args)


def train(args):
    args.max_sentence_length = args.unrolled_steps+1
    sample_file = "sample_"+time.strftime("%Y-%m-%d-%H-%M-%S")+".txt"
    reader = Reader(args.data_dir, max_vocabSize=args.vocab_size, max_sentence_length=args.max_sentence_length)
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)
    with open(os.path.join(args.save_dir, 'config.pkl'), 'wb') as f:
        pickle.dump(args, f)
    
    model = Model(args)
    session_conf = tf.ConfigProto(inter_op_parallelism_threads = args.inter_threads,
                                  intra_op_parallelism_threads = args.intra_threads,
                                  allow_soft_placement=True,
                                  log_device_placement=True)
    sess = tf.Session(config=session_conf)
    with sess.as_default() as sess:
        # instrument for tensorboard
        summaries = tf.summary.merge_all()
        writer = tf.summary.FileWriter(os.path.join(args.log_dir, time.strftime("%Y-%m-%d-%H-%M-%S")))
        writer.add_graph(sess.graph)

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())
        for epoch_idx in range(args.num_epochs):
            sess.run(tf.assign(model.lr,
                               args.learning_rate * (args.decay_rate ** epoch_idx)))
            batches = reader.batch_iter(reader.train_data, args.batch_size)
            state = sess.run(model.initial_state)
            train_loss_sum = 0.0
            b = 0
            for batch in batches:
                start = time.time()
                x, y = batch
                feed = {model.input_data: x, model.targets: y}
                c, h = model.initial_state
                feed[c] = state.c
                feed[h] = state.h
#                train_loss, state, _ = sess.run([model.cost, model.final_state, model.train_op], feed)
                # instrument for tensorboard
                summ, train_loss, state, _, predictions = sess.run([summaries, model.cost, model.final_state, model.train_op, model.predictions], feed)
                train_loss_sum += train_loss
                writer.add_summary(summ, epoch_idx * reader.num_batches_per_epoch + b)
                end = time.time()
                print("{}/{} (epoch {}), perplexity = {:.3f}, time/batch = {:.3f}"
                      .format(epoch_idx * reader.num_batches_per_epoch + b,
                              args.num_epochs * reader.num_batches_per_epoch,
                              epoch_idx,
                              train_loss,
                              end - start))
                if ((epoch_idx * reader.num_batches_per_epoch+ b) % args.save_every == 0 
                    or (epoch_idx == args.num_epochs-1 
                    and b == reader.num_batches_per_epoch-1)):
                    # save for the last result
                    checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path,
                               global_step=epoch_idx * reader.num_batches_per_epoch + b)
                    print("model saved to {}".format(checkpoint_path))
                    print("Average perplexity = {:.3f}".format(train_loss_sum/(args.unrolled_steps*(b+1))))
                    with open(os.path.join(args.save_dir, sample_file), "a") as f: # Write samples to file            
                        f.write("Epoch: {}, batch: {}/{}".format(epoch_idx, 
                                epoch_idx * reader.num_batches_per_epoch + b,
                                args.num_epochs * reader.num_batches_per_epoch))
                        f.write("\n")
                        f.write(" ".join(reader.id_to_word(predictions)))
                        f.write("\n")
                b += 1
if __name__ == '__main__':
    main()
