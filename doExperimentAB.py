import tensorflow as tf
import pickle
import numpy as np
import math

from model import Model

if __name__ == '__main__':
    sess = tf.Session()
    ckpt = tf.train.get_checkpoint_state('save')
    if ckpt and ckpt.model_checkpoint_path:
        saver = tf.train.import_meta_graph('save/model.ckpt-150.meta')
        saver.restore(sess, ckpt.model_checkpoint_path)
        xs = np.array(pickle.load(open('data/eval_data.pkl', 'rb')))
        xs = np.reshape(xs, [-1, 30])
        config = None
        with open('save/config.pkl', 'rb') as f:
            config = pickle.load(f)
        model = Model(config, testing=True)
        tf.Graph().as_default()
        tf.global_variables_initializer().run(session=sess)
        fout = open('perplexityA.out', 'w')
        for x in xs:
            feed = {model.input_data: x[:29].reshape([1, -1])}
            probs = sess.run(model.probs, feed)
            sump = 0.0; count = 0
            for i in range(0, 29):
                if x[i] == 1:
                    break;
                sump += math.log(probs[i][x[i+1]])
                count += 1
            perp = pow(2, -(sump/count))
            fout.write(str(perp)+'\n')
        fout.close()
    else:
        print 'no checkpoint found.'
    