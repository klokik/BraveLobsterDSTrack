
import time

import numpy as np
import tensorflow as tf

import cells_reader
from appconfig import *

class RnnModel(object):
  def __init__(self, config):
    self.config = config

  def buildRnnGraph(self, cell_type = "RNN"):
    config = self.config
    num_steps = config.num_steps
    num_layers = config.num_layers
    state_size = config.state_size
    batch_size = config.batch_size
    # embedding_size = config.embedding_size

    inputs = tf.placeholder(tf.float32, shape = [batch_size, num_steps, state_size], name = "input_palceholder")
    labels = tf.placeholder(tf.float32, shape = [batch_size, num_steps, state_size], name = "labels_placeholder")

    # with tf.variable_scope("my_model"):
    #   embeddings = tf.get_variable("embeddings_matrix", [num_classes, state_size])

    rnn_inputs = inputs #tf.nn.embedding_lookup(embeddings, inputs)
    # rnn_inputs = inputs

    with tf.variable_scope("my_model"):
      def make_cell(state_size, cell_type, dropout = None):
        cell = None
        if cell_type == "RNN":
          cell = tf.keras.layers.SimpleRNNCell(state_size)
        elif cell_type == "GRU":
          cell = tf.contrib.rnn.GRUCell(state_size)
        elif cell_type == "LSTM":
          cell = tf.contrib.rnn.LSTMCell(state_size, state_is_tuple = True)
        elif cell_type == "BLSTM":
          cell = tf.contrib.rnn.BasicLSTMCell(state_size, forget_bias = 0.0, state_is_tuple = True)
        else:
          raise Exception("Unknown cell_type")

        if dropout:
          cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob = dropout)

        return cell

      cell = tf.contrib.rnn.MultiRNNCell([make_cell(state_size, config.cell_type, config.dropout)
                                            for _ in range(num_layers)], state_is_tuple = True)
      init_state = cell.zero_state(batch_size, tf.float32)
      # print(init_state)

      rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, rnn_inputs, initial_state = init_state)
      # rnn_outputs, final_state = tf.scan(lambda a, x: cell(x, a[1]),
      #                                     tf.transpose(rnn_inputs, [1, 0, 2]),
      #                                     initializer = (tf.zeros([batch_size, state_size]), init_state))
      # print(final_state)


      W = tf.get_variable("W", [state_size, state_size])
      b = tf.get_variable("b", [state_size], initializer = tf.constant_initializer(0.0))

    print(rnn_outputs)
    # throw("kek")

    rnn_outputs = tf.reshape(rnn_outputs, [-1, state_size])

    rnn_outputs = tf.nn.xw_plus_b(rnn_outputs, W, b)
    rnn_outputs = tf.reshape(rnn_outputs, [-1, num_steps, state_size])
    # print(logits)

    rnn_outputs = tf.nn.leaky_relu(rnn_outputs)

    # last_prediction = rnn_outputs[] #tf.nn.softmax(logits[-1, :])
    last_prediction = rnn_outputs[-1, :, :]
    # print(last_prediction)

    # labels_flat = tf.reshape(labels, [-1, state_size])
    # loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = labels_flat, logits = logits))

    # losses = tf.contrib.seq2seq.sequence_loss(
    #   logits = logits,
    #   targets = labels,
    #   weights = tf.ones([batch_size, num_steps]),
    #   average_across_timesteps = False,
    #   average_across_batch = True)
    # loss = tf.reduce_sum(losses)

    # print(labels.shape)
    # print(rnn_outputs.shape)
    # loss = tf.losses.mean_squared_error(labels, rnn_outputs)
    # loss = 
    loss = tf.reduce_mean(tf.math.square(tf.math.log(tf.math.divide((rnn_outputs + 1), (labels + 1)))))

    with tf.variable_scope("my_model"):
      new_lrate = tf.placeholder(tf.float32, shape = [], name = "new_learning_rate_placeholder")
      l_rate = tf.get_variable("learning_rate", [])
      lr_update_op = tf.assign(l_rate, new_lrate)

    optimizer = tf.train.AdamOptimizer(config.learning_rate)
    # optimizer = tf.train.GradientDescentOptimizer(l_rate)

    trainable_vars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, trainable_vars), config.max_grad_norm)
    # train_step = optimizer.minimize(loss)
    train_step = optimizer.apply_gradients(
      zip(grads, trainable_vars),
      global_step = tf.train.get_or_create_global_step())
    
    return dict(
      inputs = inputs,
      labels = labels,
      init_state = init_state,
      final_state = final_state,
      last_prediction = last_prediction,
      loss = loss,
      train_step = train_step,
      update_lr = lr_update_op,
      new_lr = new_lrate,
      saver = tf.train.Saver(tf.global_variables())
    )

  def trainNetwork(self, graph, data, emb_reader, save = None, restore = True):
    t = time.time()

    config = self.config

    # random.seed(1337)
    # tf.set_random_seed(1337)

    with tf.Session() as sess:
      if isinstance(save, str) and restore:
        try:
          print("Restoring model from {}".format(save))
          graph["saver"].restore(sess, save)
        except Exception as e:
          print("Checkpoint not found or invalid, retraining model")
          sess.run(tf.global_variables_initializer())
      else:
        sess.run(tf.global_variables_initializer())

      total_steps = 0

      for e_id, epoch in enumerate(cells_reader.genEpochsBatches(data, config.max_epoch, config.num_steps, config.batch_size, emb_reader = emb_reader)):
        num_steps = len(epoch)
        epoch_start_time = time.time()

        lrate = config.learning_rate
        if (e_id + 1) > config.lr_decay_start:
          lrate = config.learning_rate * config.learning_rate_decay ** (e_id + 1 - config.lr_decay_start)

        sess.run([graph["update_lr"]], feed_dict = {graph["new_lr"] : lrate})
        
        loss_sum = 0
        for batch in epoch:
          xs, ys = batch
          feed_dict = { graph["inputs"] : xs, graph["labels"] : ys}
          loss_, _, _ = sess.run([graph["loss"], graph["final_state"], graph["train_step"]], feed_dict)
          loss_sum += loss_
          total_steps += 1
          # print("loss {}: {}".format(total_steps, loss_))

        wps = num_steps * config.batch_size * config.num_steps / (time.time() - epoch_start_time)
        print("Epoch {}, {} steps, {:.1f} words/s, avg loss: {}, lr: {}".format(e_id + 1, num_steps, wps, loss_sum / num_steps, lrate))

        if isinstance(save, str):
          # print("Saving model to {}".format(save))
          graph["saver"].save(sess, save)

      t_end = time.time()
      print("Model trained in {}s".format(t_end - t))

      # if isinstance(save, str):
      #   print("Saving model to {}".format(save))
      #   graph["saver"].save(sess, save) #, global_step = total_steps)

  def generateSamples(self, graph, checkpoint, num_chars, init_elems, seed = 1337):
    with tf.Session() as sess:
      np.random.seed(seed)
      # sess.run(tf.global_variables_initializer())
      graph["saver"].restore(sess, checkpoint)

      uninitialized_vars_op = tf.report_uninitialized_variables()
      uninitialized_vars = sess.run(uninitialized_vars_op)

      for var in uninitialized_vars:
        print("Uninit: {}".format(var))

      feed_items = init_elems

      cur_state = None
      # cur_elem = [init_elem]

      elems = []

      # print(feed_items)
      # for i in range(num_chars):
      for cur_elem in feed_items:
        # print(cur_elem)
        if cur_state is not None:
          feed_dict = {graph["inputs"] : [[cur_elem]], graph["init_state"] : cur_state}
        else:
          feed_dict = {graph["inputs"] : [[cur_elem]]}

        cur_state, prediction = sess.run([graph["final_state"], graph["last_prediction"]], feed_dict)

        # cur_elem = prediction
        # print(cur_elem)
        elems.append(np.squeeze(cur_elem))

      for i in range(num_chars):
        # if cur_state is not None:
        feed_dict = {graph["inputs"] : cur_elem.reshape((1, 1, -1)), graph["init_state"] : cur_state}
        # else:
        #   feed_dict = {graph["inputs"] : [cur_elem]}

        cur_state, prediction = sess.run([graph["final_state"], graph["last_prediction"]], feed_dict)

        cur_elem = prediction
        # print(cur_elem)
        elems.append(np.squeeze(cur_elem))

      result = np.array(elems)
      # print(result)

      return result

def main():
  data = cells_reader.loadCellsData("../data/cells_num.pickle")
  emb_reader = None

  config1 = \
    ModelConfig(128,
      num_layers = 4,
      batch_size = 32,
      state_size = data["num_cells"], #emb_reader.state_size,data["data"]
      # embedding_size = 200,
      # max_epoch = 256,
      cell_type = "BLSTM",
      # learning_rate = 0.005,
      dropout = None)

  model = RnnModel(config1)

  checkpoint = "/tmp/saves/" + config1.checkpointName() + ".ckpt"

  config1.train = False
  # config1.train = True

  if config1.train:
    with tf.Graph().as_default():
      t = time.time()
      graph = model.buildRnnGraph() #data["num_classes"])
      t_end = time.time()
      print("Graph built in {}s".format(t_end - t))

      # while True:
      for era in range(1, 1+1):
        print("Training era {}".format(era))
        model.trainNetwork(graph, data, emb_reader, checkpoint, restore = True)

  print("\n\nGenerating data: ")

  with tf.Graph().as_default():
    model.config.num_steps = 1
    model.config.batch_size = 1
    model.config.max_epoch = 1
    model.config.dropout = None
    graph = model.buildRnnGraph()

    predicted_traffic = model.generateSamples(graph, checkpoint, 172, data["data"][-172:], seed = 1336)
    # for var in tf.global_variables(): print(var.name)
    # print(predicted_traffic.shape)
    # print(predicted_traffic)
    import cv2
    cv2.imwrite("/tmp/unkek.png", np.maximum(predicted_traffic, np.zeros(predicted_traffic.shape)))

if __name__ == '__main__':
  main()
