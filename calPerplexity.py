import tensorflow as tf
import pickle
import numpy as np
import math

from model import Model

if __name__ == '__main__':
    config = None
    with open('save/config.pkl', 'rb') as f:
        config = pickle.load(f)
    model = Model(config, training=False)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver =  tf.train.Saver(tf.global_variables())

    ckpt = tf.train.get_checkpoint_state('save')
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)

        xs = np.array(pickle.load(open('data/eval_data.pkl', 'rb')))
        xs = np.reshape(xs, [-1, 30])

        fout = open('perplexityA.out', 'w')
        sumPerp = 0.0; ind = 0
        for x in xs:
            if ind%1000 == 0:
                print(str(ind)+' finished.')
            ind += 1
            sump = 0.0; count = 0
            state = sess.run(model.initial_state)
            for i in range(0, 29):
                if x[i] == 1:
                    break;
                feed = {model.input_data: np.array(x[i]).reshape((1, 1)), model.initial_state: state}
                prob, state = sess.run([model.probs, model.final_state], feed)
                sump += math.log(prob[0][x[i+1]])
                count += 1
            perp = pow(2, -(sump/count))
            sumPerp += perp
            fout.write(str(perp)+'\n')
        fout.close()
        print('Average perplexity: '+str(sumPerp/len(xs)))
    else:
        print ('no checkpoint found.')
