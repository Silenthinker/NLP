import inspect
import tensorflow as tf

def data_type():
    return tf.float32


class Input(object):
    """The input data."""
    
    def __init__(self, config, data, reader, name=None):  
        """
        Args:
            reader: instance of class Reader
        """
        self.batch_size = config.batch_size
        self.unrolled_steps = config.unrolled_steps
        self.epoch_size = ((len(data) // self.batch_size) - 1) // self.unrolled_steps
        self.input_data, self.targets = reader.batch_producer(
            data, self.batch_size, name=name) # input_data, targets: [batch_size, unrolled_steps]

class LanguageModel(object):
    """Language Model built based on RNN"""

    def __init__(self, is_training, config, input_):
        self.input = input_
    
        batch_size = input_.batch_size
        unrolled_steps = input_.unrolled_steps
        hidden_size = config.hidden_size
        vocab_size = config.vocab_size
        learning_rate = config.learning_rate
        def lstm_cell():
            # Check if reuse is compatible
            if 'reuse' in inspect.getargspec(
                    tf.contrib.rnn.BasicLSTMCell.__init__).args:
                return tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=0.0, state_is_tuple=True,
                                                    reuse=tf.get_variable_scope().reuse)
            else:
                return tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=0.0, state_is_tuple=True)
        cell = lstm_cell()
        self.initial_state = cell.zero_state(batch_size, data_type())
    
        embedding = tf.get_variable("embedding", 
                                    initializer=tf.random_uniform_initializer(-config.init_scale, config.init_scale),
                                    shape=[vocab_size, hidden_size], 
                                    dtype=data_type())
        inputs = tf.nn.embedding_lookup(embedding, input_.input_data) # [None, unrolled_size, embedding_size]
        # Implemented RNN cell, unrolled to feedforward
        outputs = []
        state = self.initial_state
        # RNN cell
        with tf.variable_scope("RNN"):
            for time_step in range(unrolled_steps):
                if time_step > 0: tf.get_variable_scope().reuse_variables() # make sure use one weight
                (cell_output, state) = cell(inputs[:, time_step, :], state) # [batch_size, hidden_size]
                outputs.append(cell_output)
    
        output = tf.reshape(tf.concat(axis=1, values=outputs), [-1, hidden_size]) # [batch_size*unrolled_steps, hidden_size]
#        print("---------")
#        print(output.get_shape())
#        print("---------")
        softmax_w = tf.get_variable("softmax_w", 
                                    shape=[hidden_size, vocab_size], 
                                    initializer=tf.contrib.layers.xavier_initializer(),
                                    dtype=data_type())
        softmax_b = tf.get_variable("softmax_b", 
                                    initializer=tf.zeros_initializer(),
                                    shape=[vocab_size], 
                                    dtype=data_type())
        logits = tf.matmul(output, softmax_w) + softmax_b # [batch_size*unrolled_steps, vocab_size]
#        print("---------")
#        print(logits.get_shape())
#        print(input_.targets.get_shape())
#        print("---------")
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=tf.reshape(input_.targets, [-1]),
                logits=logits
                )
        self.cost = tf.reduce_mean(losses)
        # Compute perplexity
        # TODO: The <eos> symbol is part of the sequence, while the <pad> symbols (if any) are not.
        seq_loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
                [logits],
                [tf.reshape(input_.targets, [-1])],
                 [tf.ones([batch_size * unrolled_steps], dtype=data_type())])
        self.perp_all = tf.reduce_sum(seq_loss) / batch_size
                                     
        self.final_state = state
        # TODO: maybe sampling?
#        self.predictions = tf.reshape(tf.cast(tf.multinomial(logits, 1, name="prediction"), "int32"), [-1, unrolled_steps])
        self.predictions = tf.reshape(tf.argmax(logits, 1, name="predictions"), [-1, unrolled_steps])
        correct_predictions = tf.equal(tf.cast(tf.reshape(self.predictions, [-1]), "int32"), tf.reshape(input_.targets, [-1]))
        self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
        if not is_training:
            return
        
        # Define training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False) # number of batches seen by the graph
        optimizer = tf.train.AdamOptimizer(learning_rate) 
        grads_and_vars = optimizer.compute_gradients(self.cost)
        gradients, variables = zip(*grads_and_vars)
        gradients_clipped, _ = tf.clip_by_global_norm(gradients, config.max_grad_norm)
        self.train_op = optimizer.apply_gradients(zip(gradients_clipped, variables), global_step=global_step)
    