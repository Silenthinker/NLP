import inspect

import numpy as np

import tensorflow as tf
#from tensorflow.contrib import legacy_seq2seq



class Model():
    def __init__(self, args, training=True):
        self.args = args
        if not training: # if not training, sample one by one instead of batch mode
            args.batch_size = 1
            args.unrolled_steps = 1

        def lstm_cell():
            """
            A wrapper for compatibility
            """
            if 'reuse' in inspect.getargspec(
                tf.contrib.rnn.BasicLSTMCell.__init__).args:
                    return tf.contrib.rnn.BasicLSTMCell(
                        args.hidden_size, input_size=args.embedding_size, forget_bias=0.0, state_is_tuple=True,
                        reuse=tf.get_variable_scope().reuse)
            else:
                return tf.contrib.rnn.BasicLSTMCell(
                        args.hidden_size, input_size=args.embedding_size, forget_bias=0.0, state_is_tuple=True)
                
        def loop(prev, _):
            """
            If not training, predict next word based on previous predicted word
            """
            prev = tf.matmul(prev, softmax_w) + softmax_b
            prev_symbol = tf.stop_gradient(tf.argmax(prev, 1)) # Stops gradient computation.
            return tf.nn.embedding_lookup(embedding, prev_symbol)

        def rnn(inputs_, initial_state, cell, loop_function=None, scope=None):
            """
            RNN loop
            """
            with tf.variable_scope("rnn"):
                state = initial_state
                outputs = []
                prev = None
                for i, input_ in enumerate(inputs_):
                    if loop_function is not None and prev is not None:
                        input_ = loop_function(prev, i)
                    if i > 0:
                        tf.get_variable_scope().reuse_variables() # make sure reuse weight...
                    output, state = cell(input_, state)
                    outputs.append(output)
                    if loop_function is not None:
                        prev = output
            return outputs, state

        # input and target data
        self.input_data = tf.placeholder(tf.int32, [args.batch_size, args.unrolled_steps])
        self.targets = tf.placeholder(tf.int32, [args.batch_size, args.unrolled_steps])
        
        # lstm cell
        self.cell = cell = lstm_cell()
        # initial state
        self.initial_state = cell.zero_state(args.batch_size, tf.float32)
        
        # embedding layer
        embedding = tf.get_variable("embedding", 
                                    initializer=tf.random_uniform_initializer(-args.init_scale, args.init_scale), 
                                    shape=[args.vocab_size, args.embedding_size])
        inputs = tf.nn.embedding_lookup(embedding, self.input_data) # [batch_size, unrolled_steps, embedding_size]
        inputs = tf.split(inputs, args.unrolled_steps, 1) # list of length unrolled_steps, each element: [batch_size, 1, embedding_size]
        inputs = [tf.squeeze(input_, [1]) for input_ in inputs]
        
        # rnn
        outputs, last_state = rnn(inputs, self.initial_state, cell, loop_function=loop if not training else None, scope='softmax')
        output = tf.reshape(tf.concat(outputs, 1), [-1, args.hidden_size]) # [batch_size*unrolled_steps, hidden_size]
        
        # softmax
        with tf.variable_scope('softmax'):
            softmax_w = tf.get_variable("softmax_w",
                                        initializer=tf.contrib.layers.xavier_initializer(),
                                        shape=[args.hidden_size, args.vocab_size])
            softmax_b = tf.get_variable("softmax_b", [args.vocab_size])
        self.logits = tf.matmul(output, softmax_w) + softmax_b # [batch_size*unrolled_steps, vocab_size]
        self.predictions = tf.reshape(tf.argmax(self.logits, 1, name="predictions"), [-1, args.unrolled_steps])
        probs_distr = tf.nn.softmax(self.logits) # same shape as logits
        self.probs = tf.argmax(probs_distr, axis=1, name="compute_word_prob")
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(self.targets, [-1]),
                                                              logits=self.logits)
        # TODO: compute sentence-level perplexity without considering <pad>
#        loss = legacy_seq2seq.sequence_loss_by_example([self.logits],
#                                                       [tf.reshape(self.targets, [-1])],
#                                                        [tf.ones([args.batch_size * args.unrolled_steps])])
        with tf.name_scope('cost'):
            self.cost = tf.reduce_sum(loss) / args.batch_size # sentence level cross entropy
                                     
        self.final_state = last_state
        
        if not training:
            return
        
        # gradient
        self.lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
                args.grad_clip)
        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))
    
    def sample(self, sess, id_toWord, word_toId, num=20, beg=['i']):
        def word_to_id(w):
            return word_toId[w] if w in word_toId else word_toId['<unk>']
        
        state = sess.run(self.cell.zero_state(1, tf.float32))
        beg.insert(0, '<eos>') # add <eos>
        # first feed given words
        for w in beg[:-1]:
            x = np.zeros((1, 1))
            x[0, 0] = word_to_id(w) #word_toId[w]
            feed = {self.input_data: x, self.initial_state: state}
            [state] = sess.run([self.final_state], feed)
        res = beg
        w = beg[-1]
        # from last word, iteratively generate next word
        for n in range(num-1):
            x = np.zeros((1, 1)) # same rank required
            x[0, 0] = word_to_id(w) # word_toId[w]
            feed = {self.input_data: x, self.initial_state: state}
            [probs, state] = sess.run([self.probs, self.final_state], feed)
            # probs: [1, vocab_size]
            sample = np.argmax(probs) # return scalar
            pred = id_toWord[sample]
            res.append(pred)
            w = pred
            if w == '<eos>':
                break
        return " ".join(res)
