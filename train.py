import tensorflow as tf
import argparse
import time
import os
import pickle
import numpy as np
from gensim import models
from data_utils import Reader
from model import Model

def configure(args):
    args.hidden_size = 512
    args.projector_size = args.hidden_size
    args.learning_rate = 0.01
    config = args.config
    if config == "A":
        args.pretrain = False
    elif config == "B":
        args.pretrain = True
    else:
        args.hidden_size = 1024
        args.pretrain = True
        args.projector_size = 512
        args.learning_rate = 0.001
    return args

def calSingleBatchPerp(x, y, sess, model):
    """
    Compute average sentence-level perplexity per batch
    Args:
        x: [batch_size, unrolled_steps]
        y: [batch_size, unrolled_steps]
    """
    batch_size, unrolled_steps = x.shape
    sump = 0.0
    state = sess.run(model.initial_state)
    feed = {model.input_data: x, model.initial_state: state}
    prob = sess.run([model.probs], feed) # list of size 1, element: [batch_size*unrolled_steps, vocab_size]
    prob = prob[0].reshape((batch_size, unrolled_steps, -1)) # [batch_size, unrolled_steps, vocab_size]
    for i in range(batch_size):
        singlep = 0.0
        count = 0
        for j in range(unrolled_steps):
            if y[i, j] == 1: # <eos>
                break
            if y[i, j] != 2: # not <pad>
                singlep += np.log(prob[i, j, y[i, j]])
                count += 1
        singlep = np.power(2, -singlep/count)
#        print(singlep)
        sump += singlep
    return sump/batch_size

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # model
    parser.add_argument('--vocab_size', type=int, default=20000, help='size of vocabulary')
    parser.add_argument('--hidden_size', type=int, default=512, help='size of RNN hidden state')
    parser.add_argument('--embedding_size', type=int, default=100, help='size of embedding')
    parser.add_argument('--unrolled_steps', type=int, default=29, help='RNN unrolled length')
    parser.add_argument('--grad_clip', type=float, default=10., help='clip gradients at this value')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.97, help='decay rate for learning rate')
    parser.add_argument('--init_scale', type=float, default=0.1, help='initial scale for random uniform initializer')
    parser.add_argument('--config', type=str, required=True, help='(string) choose configuration of model, values: A, B, C')
    # training
    parser.add_argument('--pretrain', type=bool, required=True, help='(bool) pretrain word embedding; True for pretrain; False otherwise')
    parser.add_argument('--batch_size', type=int, default=64, help='minibatch size')
    parser.add_argument('--num_epochs', type=int, default=3, help='number of epochs')
    parser.add_argument('--inter_threads', type=int, default=4, help='inter_op_parallelism_threads')
    parser.add_argument('--intra_threads', type=int, default=4, help='intra_op_parallelism_threads')
    # save and restore
    parser.add_argument('--data_dir', type=str, default='data/', help='data directory containing training, evaluation, and continuation data')
    parser.add_argument('--save_dir', type=str, default='save', help='directory to store checkpointed models')
    parser.add_argument('--log_dir', type=str, default='logs', help='directory to store tensorboard logs')
    parser.add_argument('--save_every', type=int, default=200, help='save frequency')
    args = parser.parse_args()
    tf.set_random_seed(15) # allow reproduce results
    configure(args)
    train(args)

def load_embedding(session, data_dir, vocab_size, embedding_size, word_toId, emb):
    '''
    Args:
        args: parsed argument that path to embedding file and dimensionality of the external embedding
        session: Tensorflow session object
        word_toId: A dictionary mapping token strings to vocabulary IDs
        emb: Embedding tensor of shape vocabulary_size x dim_embedding      
    '''
    # TODO: better learning rate management
    path = os.path.join(data_dir, "wordembeddings-dim100.word2vec")
    print("Loading external embeddings from %s" % path)
    model = models.KeyedVectors.load_word2vec_format(path, binary=False)
    external_embedding = np.zeros(shape=(vocab_size, embedding_size))
    matches = 0
    for tok, idx in word_toId.items():
        if tok in model.vocab:
            external_embedding[idx] = model[tok]
            matches += 1
        else:
            print("%s not in embedding file" % tok)
            external_embedding[idx] = np.random.uniform(low=-0.25, high=0.25, size=embedding_size)
        
    print("%d words out of %d could be loaded" % (matches, vocab_size))
    
    pretrained_embeddings = tf.placeholder(tf.float32, [None, None]) # even if emb is not a placeholder
    assign_op = emb.assign(pretrained_embeddings)
    session.run(assign_op, {pretrained_embeddings: external_embedding})
    
def train(args):
    """
    Args:
        args: parsed arguments
    """
    args.max_sentence_length = args.unrolled_steps+1
    dev_perp_file = "dev_perp_"+time.strftime("%Y-%m-%d-%H-%M-%S")+".txt"
    reader = Reader(args.data_dir, max_vocabSize=args.vocab_size, max_sentence_length=args.max_sentence_length)
    num_batches_per_epoch = int(len(reader.train_data)/args.batch_size)
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)
    with open(os.path.join(args.save_dir, 'config.pkl'), 'wb') as f:
        pickle.dump(args, f)
    model = Model(args, training=True)
    # for tensorboard
    tf.summary.histogram('logits', model.logits)
    tf.summary.scalar('train_loss', model.cost)
    session_conf = tf.ConfigProto(inter_op_parallelism_threads = args.inter_threads,
                                  intra_op_parallelism_threads = args.intra_threads,
                                  allow_soft_placement=True,
                                  log_device_placement=True)
    sess = tf.Session(config=session_conf)
    min_avg_dev_perp = np.inf
    with sess.as_default() as sess:
        # instrument for tensorboard
        summaries = tf.summary.merge_all()
        writer = tf.summary.FileWriter(os.path.join(args.log_dir, time.strftime("%Y-%m-%d-%H-%M-%S")))
        writer.add_graph(sess.graph)
        sess.run(tf.global_variables_initializer())
        if args.pretrain:
            load_embedding(sess, args.data_dir, args.vocab_size, args.embedding_size, reader.word_toId, model.embedding)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
        for epoch_idx in range(args.num_epochs):
            # TODO: better management of learning rate
            sess.run(tf.assign(model.lr,
                               args.learning_rate * (args.decay_rate ** epoch_idx)))
            batches = reader.batch_iter(reader.train_data, args.batch_size)
            train_loss_sum = 0.0
            b = 0
            for batch in batches:
                start = time.time()
                state = sess.run(model.initial_state)
                x, y = batch
                feed = {model.input_data: x, model.targets: y}
                c, h = model.initial_state
                feed[c] = state.c
                feed[h] = state.h
#                train_loss, state, _ = sess.run([model.cost, model.final_state, model.train_op], feed)
                # instrument for tensorboard
                summ, train_loss, state, _, predictions = sess.run([summaries, model.cost, model.final_state, model.train_op, model.predictions], feed)
                train_loss_sum += train_loss
                writer.add_summary(summ, epoch_idx * num_batches_per_epoch + b)
                end = time.time()
                print("{}/{} (epoch {}), average loss = {:.3f}, time/batch = {:.3f}"
                      .format(epoch_idx * num_batches_per_epoch + b,
                              args.num_epochs * num_batches_per_epoch,
                              epoch_idx,
                              train_loss_sum/(b+1),
                              end - start))
                if ((epoch_idx * num_batches_per_epoch+ b) % args.save_every == 0 
                    or (epoch_idx == args.num_epochs-1 
                    and b == num_batches_per_epoch-1)):
                    dev_batches = reader.batch_iter(reader.test_data, shuffle=False)
                    sump = 0.0
                    count = 0.0
                    for dev_batch in dev_batches:
                        dev_x_batch, dev_y_batch = dev_batch
                        sump += calSingleBatchPerp(dev_x_batch, dev_y_batch, sess, model)
                        count += 1
                    avg_dev_perp = sump / count
                    print(avg_dev_perp)
                    if avg_dev_perp < min_avg_dev_perp:
                        min_avg_dev_perp = avg_dev_perp
                        # save for the last result
                        checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')
                        saver.save(sess, checkpoint_path,
                                   global_step=epoch_idx * num_batches_per_epoch + b)
                        print("model saved to {}".format(checkpoint_path))
                    with open(os.path.join(args.save_dir, dev_perp_file), "a") as f: # Write samples to file            
                        f.write("Epoch: {}, batch: {}/{}, avg perp: {:3f}, min avg perp: {:3f}\n".format(epoch_idx, 
                                epoch_idx * num_batches_per_epoch + b,
                                args.num_epochs * num_batches_per_epoch,
                                avg_dev_perp,
                                min_avg_dev_perp))
                b += 1
if __name__ == '__main__':
    main()
