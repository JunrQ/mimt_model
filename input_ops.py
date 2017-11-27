import os
import pickle
import csv
import numpy as np
from scipy.misc import imread
import skimage.transform
import skimage.util
import math

def csvfile2iterator(csvfile_path, parent_path):
  """Given a csvfile path, return a iterator
      used for csvfile.csv in E:csvfile.csv
      e.g. AlkB1,['131902_s.bmp'],"['maternal', 'ubiquitous']"

      with element: {'gene stage': gene stage,
                 'urls': urls list,
                 'labels': label list}
  """
  with open(csvfile_path, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
      urls = []
      gene_stage = row[0]
      labels = []
      # print(row)
      for ele in row[2][1:-1].split(','):
        # print(ele)
        labels.append(ele.split('\'')[1])

      for ele in row[1][1:-1].split():
        urls.append(os.path.join(parent_path, ele.split('\'')[1]))

      yield {'gene stage': gene_stage,
             'urls': urls,
             'labels': labels}

def annot2vec(label, vocab):
  """return one-hot
  vocab shoule be list
  """
  label_list = np.zeros(len(vocab), dtype=np.int)
  for w in label:
    try:
      idx = vocab.index(w)
      label_list[idx] = 1
    except ValueError:
      continue

  return label_list

def read_image_from_single_file(filename, dtype='float64', redownload=False, try_times=2):
  """
  Args:
      filename: a path of an image
  Return:
      im: numpy.ndarray, dtype: float32
  """
  flag = True
  for tmp in range(try_times):
    try:
      im = imread(filename)
      # print(im.shape)
      flag = True
      break
    except IOError:
      flag = False

      continue

  if not flag:
    # print('[Error: %s do not exist!]' % filename)
    return -1
  else:
    im = skimage.util.img_as_float(im).astype(dtype)
    return im

def get_image_from_urls_list_concat_dim0(urls_list,
                                         ignore_diff_size=False,
                                         shape=None,
                                         dtype='float64'):
  """
  return images concat dim 0
  """
  if shape is None:
    print("Wrong shape parameters, shape should not be None.")

  first = -1
  for idx in range(0, len(urls_list)):
    im = read_image_from_single_file(urls_list[idx], dtype=dtype)
    if isinstance(im, int):
      continue

    # print(im.shape)
    if im.shape != shape:
      if ignore_diff_size:
        continue
      else:
        im = skimage.transform.resize(im, shape)
        first = idx
        break
    else:
      first = idx
      break
  # print(im.shape)
  if first == len(urls_list):
    return im
  if first == -1:
    # print("All urls failed.")
    return -1

  for idx in range(first, len(urls_list)):
    temp = read_image_from_single_file(urls_list[idx], dtype=dtype)
    if isinstance(temp, int):
      continue
    if temp.shape != shape:
      if ignore_diff_size:
        continue
      else:
        temp = skimage.transform.resize(temp, shape)

    im = np.concatenate((im, temp))
  return im

def load_data(config):
  """
  Output:
    self.raw_dataset: should be a list of dictionary whose element
                      should be {'filename':
                                 'label_index':}
  Usage:
    filenames = [d['filename'] for d in data]
    label_indexes = [d['label_index'] for d in data]
  """
  # if exist, get it.
  if os.path.exists(config.RAW_DATASET_PATH):
    with open(config.RAW_DATASET_PATH, 'rb') as f:
      vocab = pickle.load(f)
      raw_dataset = pickle.load(f)

    with open(config.VALID_DATASET_PATH, 'rb') as f:
      valid_dataset = pickle.load(f)
    with open(config.TEST_DATASET_PATH, 'rb') as f:
      test_dataset = pickle.load(f)
  else:
    DATASET_ITERATOR = csvfile2iterator(csvfile_path=config.CSVFILE_PATH, parent_path=config.IMAGE_PARENT_PATH)
    raw_dataset = []

    tmp_dataset = []
    for ele in DATASET_ITERATOR:
      gene_stage = ele['gene stage']
      urls_list = ele['urls']
      label = ele['labels']

      # choose stage
      stage = int(gene_stage[-1])
      if stage not in config.stages:
        continue
      #
      if len(urls_list) > config.max_sequence_length:
        continue
      #
      if len(label) < config.annot_min_per_group:
        continue

      # if don not have only_word in label, continue
      tmp_flag = True
      if not config.only_word is None:
        tmp_flag = True
        for _word_ele in config.only_word:
          if _word_ele not in label:
            tmp_flag = False
      if not tmp_flag:
        continue

      # remove self.deprecated_word
      if config.deprecated_word is not None:
        tmp_label = []
        for _tmp in label:
          if _tmp in config.deprecated_word:
            continue
          else:
            tmp_label.append(_tmp)
        label = tmp_label

      tmp_dataset.append({'urls': urls_list,
                          'annot': label,
                          'gene stage': gene_stage})

    tmp_list = []
    for ele in tmp_dataset:
      label = ele['annot']
      tmp_list += label
    tmp_list = np.array(tmp_list)
    tmp_vocab, tmp_vocab_count = np.unique(tmp_list, return_counts=True)

    # only the top k labels
    tmp_dataset_0 = tmp_dataset
    if not config.annotation_number is None:
      max_arg = np.argsort(tmp_vocab_count)
      top_k_labels_idx = max_arg[-config.annotation_number:]
      allowed_word = tmp_vocab[top_k_labels_idx]
      tmp_dataset_1 = []
      for ele in tmp_dataset_0:
        label = ele['annot']
        tmp_label = []
        for ele_word in label:
          if ele_word in allowed_word:
            tmp_label.append(ele_word)
        if len(tmp_label) > 0:
          tmp_dataset_1.append({'urls': ele['urls'],
                                'annot': tmp_label,
                                'gene stage': ele['gene stage']})

    else:
      tmp_dataset_0 = []
      tmp_vocab = list(tmp_vocab)
      tmp_vocab_count = list(tmp_vocab_count)
      for tmp in tmp_dataset:
        urls = tmp['urls']
        annot = tmp['annot']
        gene_stage = tmp['gene stage']
        annot_new = []
        for tmp_label in annot:
          tmp_idx = tmp_vocab.index(tmp_label)
          if tmp_vocab_count[tmp_idx] >= config.min_annot_num:
            annot_new.append(tmp_label)
        if len(annot_new) < config.annot_min_per_group:
          continue
        tmp_dataset_0.append({'urls': urls,
                              'annot': annot_new,
                              'gene stage': gene_stage})

      tmp_dataset_1 = tmp_dataset_0

      # Done filters input data

    # build vocab
    tmp_list = []
    for ele in tmp_dataset_1:
      tmp_list += ele['annot']
    tmp_list = np.array(tmp_list)
    vocab = list(np.unique(tmp_list))

    raw_dataset = []
    for ele in tmp_dataset_1:
      urls = ele['urls']
      tmp_urls = []
      for url in urls:
        if os.path.isfile(url):
          tmp_urls.append(url)
      if len(tmp_urls) <= 0:
        continue
      else:
        raw_dataset.append({'urls': tmp_urls,
                            'annot': ele['annot'],
                            'gene stage': ele['gene stage']})

    np.random.shuffle(raw_dataset)
    valid_num = math.ceil(len(raw_dataset) * config.proportion['val'])
    test_num = math.ceil(len(raw_dataset) * config.proportion['test'])
    valid_dataset = raw_dataset[:valid_num]
    test_dataset = raw_dataset[(valid_num + 1):(valid_num + test_num)]
    raw_dataset = raw_dataset[(valid_num + test_num + 1):]

    with open(config.RAW_DATASET_PATH, 'wb') as f:
      pickle.dump(vocab, f, True)
      pickle.dump(raw_dataset, f, True)

    with open(config.VALID_DATASET_PATH, 'wb') as f:
      pickle.dump(valid_dataset, f, True)

    with open(config.TEST_DATASET_PATH, 'wb') as f:
      pickle.dump(test_dataset, f, True)

  return vocab, raw_dataset, valid_dataset, test_dataset

def load_data_0(csvfile, parent_path):
  with open(csvfile, 'r') as f:
    reader = csv.reader(f)
    _ = reader.__next__()
    for row in reader:
      gene = row[0]
      stage = int(row[1])
      labels = []
      urls = []
      for ele in row[2][2:-2].split(','):
        for tmp_name in ele.split('\''):
          if tmp_name.endswith('.bmp'):
            tmp_url = os.path.join(parent_path, tmp_name)
            # print(tmp_url)
            urls.append(tmp_url)
      for ele in row[3][1:-1].split(','):
        if(len(ele)) < 3:
          continue
        tmp_word = ele.split('\'')[1]
        # print(tmp_word)
        labels.append(tmp_word)
      yield {'gene stage': gene+str(stage),
             'urls': urls,
             'annot': labels}

class Dataset(object):
  """Dataset class."""
  def __init__(self,
               config):
    """Build input dataset."""
    vocab, raw_dataset, valid_dataset, test_dataset = load_data(config)
    self.vocab = vocab
    self.raw_dataset = raw_dataset
    self.valid_dataset = valid_dataset
    self.test_dataset = test_dataset

class Dataset_0(object):
  def __init__(self, config, parent_path='/home/litiange/prp_file'):
    annot_num = config.annotation_number
    self.train_dataset = []
    self.test_dataset = []

    tmp_csv = str(annot_num)+'_'+'labels_train.csv'
    tmp_path = os.path.join(parent_path, tmp_csv)
    for ele in load_data_0(tmp_path, parent_path=config.IMAGE_PARENT_PATH):
      urls = []
      for url in ele['urls']:
        if os.path.isfile(url):
          urls.append(url)
      if len(urls) == 0:
        continue
      self.train_dataset.append({'urls': urls,
                          'annot': ele['annot'],
                          'gene stage': ele['gene stage']})

    tmp_csv = str(annot_num) + '_' + 'labels_train.csv'
    tmp_path = os.path.join(parent_path, tmp_csv)
    for ele in load_data_0(tmp_path, parent_path=config.IMAGE_PARENT_PATH):
      urls = []
      for url in ele['urls']:
        if os.path.isfile(url):
          urls.append(url)
      if len(urls) == 0:
        continue
      self.test_dataset.append({'urls': urls,
                          'annot': ele['annot'],
                          'gene stage': ele['gene stage']})

    np.random.shuffle(self.train_dataset)

    train_num = math.ceil(len(self.train_dataset) * 0.8)
    self.raw_dataset = self.train_dataset[:train_num]
    self.valid_dataset = self.train_dataset[(train_num+1):]

    self.vocab = []
    vocab_dict = {}

    for ele in self.train_dataset:
      for word in ele['annot']:
        if not (word in self.vocab):
          self.vocab.append(word)
          vocab_dict[word] = 1
        else:
          vocab_dict[word] += 1
    for ele in self.test_dataset:
      for word in ele['annot']:
        if not (word in self.vocab):
          self.vocab.append(word)
          vocab_dict[word] = 1
        else:
          vocab_dict[word] += 1

    total_num = np.sum(vocab_dict.values())

    tmp_raw_dataset = []

    for ele in self.raw_dataset:
      tmp_count = 0
      for tmp_word in ele['annot']:
        tmp_count += vocab_dict[tmp_word]

      time_to_add = math.ceil(total_num * 1.0 / tmp_count)
      random_time_to_add = np.random.randint(1, time_to_add+1)
      tmp_raw_dataset += (ele * random_time_to_add)

    self.raw_dataset = tmp_raw_dataset
    np.random.shuffle(self.raw_dataset)




