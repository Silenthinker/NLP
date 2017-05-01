import time
import os
import numpy as np
import tensorflow as tf

from data_utils import Reader
from LanguageModel import Input, LanguageModel

## Define parameters

flags = tf.flags
logging = tf.logging
flags.DEFINE_string("data_path", "data", "Where the training/test data is stored.")
flags.DEFINE_string("save_path", "save", "Model output directory.")
FLAGS = flags.FLAGS

FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
  print("{}={}".format(attr.upper(), value))
print("")

class TrainConfig(object):
    """Train config."""
    learning_rate = 0.1
    init_scale = 0.1
    max_grad_norm = 10
    unrolled_steps = 29
    max_sentence_length = unrolled_steps + 1
    hidden_size = 512
    max_epoch = 4
    max_max_epoch = 13
    batch_size = 64
    vocab_size = 10000# TODO: 20000
    
def run_epoch(session, model, eval_op=None, verbose=False):
    """Runs the model on the given data."""
    start_time = time.time()
    perp_all = 0.0
    iters = 0
    perp_raw = 0.0
    accuracy = 0.0
    predictions = []
    state = session.run(model.initial_state)


    fetches = {
            "cost": model.cost,
            "final_state": model.final_state,
            "predictions": model.predictions,
            "perp_raw": model.perp_raw,
            "accuracy": model.accuracy}
    if eval_op is not None:
        fetches["eval_op"] = eval_op

    for step in range(model.input.epoch_size):
        feed_dict = {}
        c, h = model.initial_state
        feed_dict[c] = state.c
        feed_dict[h] = state.h
        vals = session.run(fetches, feed_dict)
#        cost = vals["cost"]
        state = vals["final_state"]
        predictions = vals["predictions"]
        perp_raw = vals["perp_raw"]
        accuracy = vals["accuracy"]
        perp_all += perp_raw
        iters += model.input.unrolled_steps
        print("%.3f accuracy: %.3f" % (step * 1.0 / model.input.epoch_size, accuracy))
        if verbose and step % (model.input.epoch_size // 10) == 10:
            print("%.3f perplexity: %.3f accuracy: %.3f speed: %.0f wps" %
                  (step * 1.0 / model.input.epoch_size, 
                   np.exp(perp_all / iters),
                   accuracy,
                   iters * model.input.batch_size / (time.time() - start_time)))
    return np.exp(perp_all / iters), predictions

def main(_):
    cwd = os.getcwd()
    config = TrainConfig()
    reader = Reader(max_vocabSize=config.vocab_size, max_sentence_length=config.max_sentence_length)
    train_data, eval_data = reader.raw_data(FLAGS.data_path)
    train_data = train_data[0:100000] # TODO: change train_data size

    with tf.name_scope("Train"):
        train_input = Input(config=config, data=train_data, reader=reader, name="TrainInput")
        with tf.variable_scope("Model", reuse=None):
            m = LanguageModel(is_training=True, config=config, input_=train_input)
            tf.summary.scalar("Training_Loss", m.cost)
#            tf.summary.scalar("Learning Rate", m.lr)
    
#    session_conf = tf.ConfigProto(
#            inter_op_parallelism_threads = 4,
#            intra_op_parallelism_threads = 4,
#            allow_soft_placement=True,
#            log_device_placement=True)
#    sess = tf.Session(config=session_conf)
#    saver = tf.train.Saver()
#    init_op = tf.global_variables_initializer()
#    print("done")
#    with sess.as_default() as session: # Use the with keyword to specify that calls to Operation.run() or Tensor.eval() should be executed in this session.
    sv = tf.train.Supervisor(logdir=FLAGS.save_path)
    with sv.managed_session() as session:
#        sess.run(init_op)
#        sess.graph.finalize() # Finalizes this graph, making it read-only.
        if not os.path.exists(cwd+"/sample"):
            os.makedirs(cwd+"/sample")
        sample_path = cwd+"/sample/samples_"+str(time.time())+".txt"
        for i in range(config.max_max_epoch):
            train_perplexity, predictions = run_epoch(session, m, eval_op=m.train_op,
                                     verbose=True)
            with open(sample_path, "a") as f: # Write samples to file            
                f.write("Epoch %s \n" % (i))
                f.write(" ".join(reader.id_to_word(predictions)))
                f.write("\n")
            print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))

        if FLAGS.save_path:
            print("Saving model to %s." % FLAGS.save_path)
            sv.saver.save(session, FLAGS.save_path)

if __name__ == "__main__":
    tf.app.run()