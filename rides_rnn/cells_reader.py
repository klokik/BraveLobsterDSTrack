
import random

import numpy as np
import pickle


def loadCellsData(fname):
  with open(fname, "rb") as f:
    data = pickle.load(f)

  print("Refined data length {}".format(len(data)))
  print(data.shape)

  return dict(
    data = data,
    num_cells = data.shape[1],
    pad = 0,
  )

def chunks(data, chunk_size, short_last = False, pad = None):
  for i in range(0, len(data), chunk_size):
    chunk = data[i : i + chunk_size]

    pad_size = chunk_size - len(chunk)
    if pad_size:
      if not short_last:
        if pad is not None:
          # chunk += [pad] * pad_size
          chunk = np.concatenate((chunk, np.zeros((pad_size, 38))))
        else:
          return
      # else:
      #   return

    yield chunk

def genEpochsBatches(data, epochs, num_steps, batch_size, emb_reader = None):
  # pad = data["pad_id"]
  pad = data["pad"]
  xs_chunks = chunks(data["data"][:-1], num_steps, pad = pad)
  ys_chunks = chunks(data["data"][1:], num_steps, pad = pad)

  data_chunks = list(zip(xs_chunks, ys_chunks))

  for ep in range(epochs):
    random.shuffle(data_chunks)

    batches = list(chunks(data_chunks, batch_size, short_last = False))
    
    batches_reshaped = []

    for batch in batches:
      xs = []
      ys = []
      for x, y in batch:
        xs.append(x)
        ys.append(y)
      #   xs.append(list(map(word2id.get, x)))
      #   ys.append(list(map(word2id.get, y)))

      if emb_reader:
        xs = emb_reader.lookupBatch(xs)
        ys = emb_reader.lookupBatch(ys)

      batches_reshaped.append((np.array(xs), np.array(ys)))

    # print(batches_reshaped)
    yield batches_reshaped
