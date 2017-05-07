import tensorflow as tf
import pickle
import numpy as np
import math
import argparse
import os
from model import Model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--save_dir', type=str, default='save', help='model directory to store checkpointed models')
    parser.add_argument('--data_dir', type=str, default='data/', help='data directory containing training, evaluation, and continuation data')
    parser.add_argument('--res_dir', type=str, default='res', help='data directory containing results')
    parser.add_argument('--output_file', type=str, required=True, help='output file name')
    args = parser.parse_args()
    
    config = None
    with open(os.path.join(args.save_dir, 'config.pkl'), 'rb') as f:
        config = pickle.load(f)
    model = Model(config, training=False)
    if not os.path.isdir(args.res_dir):
        os.makedirs(args.res_dir)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver =  tf.train.Saver(tf.global_variables())

    ckpt = tf.train.get_checkpoint_state(args.save_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)

        xs = np.array(pickle.load(open(os.path.join(args.data_dir, 'test_data.pkl'), 'rb')))
        xs = np.reshape(xs, [-1, 30])

        fout = open(os.path.join(args.res_dir, args.output_file), 'w')
        sumPerp = 0.0; ind = 0
        for x in xs:
            if ind%200 == 0:
                print("{}/{} finished".format(ind, len(xs)))
            ind += 1
            sump = 0.0; count = 0
            state = sess.run(model.initial_state)
            for i in range(0, 29):
                if x[i] == 1:
                    break;
                feed = {model.input_data: np.array(x[i]).reshape((1, 1)), model.initial_state: state}
                prob, state = sess.run([model.probs, model.final_state], feed)
                if x[i+1] != 2:
                    sump += math.log(prob[0][x[i+1]])
#                    print(prob[0][x[i+1]])
                    count += 1
            perp = pow(2, -(sump/count))
            sumPerp += perp
            fout.write(str(round(perp, 3))+'\n')
        fout.close()
        print('Average perplexity: {}'.format(sumPerp/len(xs)))
    else:
        print('no checkpoint found.')
