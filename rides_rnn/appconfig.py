

class ModelConfig(object):
  def __init__(self, num_steps,
                num_layers = 1,
                cell_type = "RNN",
                state_size = 32,
                embedding_size = 300,
                batch_size = 32,
                max_epoch = 10000,
                learning_rate = 1e-3,
                dropout = 0.9):
    self.num_steps = num_steps
    self.num_layers = num_layers
    self.cell_type = cell_type
    self.state_size = state_size
    self.embedding_size = embedding_size
    self.batch_size = batch_size
    self.max_epoch = max_epoch

    self.train = False
    self.learning_rate = learning_rate
    self.learning_rate_decay = 0.99
    self.lr_decay_start = 10000
    self.max_grad_norm = 5
    self.dropout = dropout

  def checkpointName(self):
    return "Drives" + "_".join(map(str, [
      self.cell_type,
      self.num_steps,
      self.num_layers,
      self.state_size,
      self.batch_size,
      # self.learning_rate
    ]))

