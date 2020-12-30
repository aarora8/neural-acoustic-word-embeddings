from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import re
import numpy as np
from collections import Counter
from kaldi_io import read_mat_scp
import logging

class Dataset(object):
    """Creat data class."""

    def __init__(self, partition, config, feature_mean=None):
        """Initialize dataset."""
        self.is_train = (partition == "train")

        self.feature_dim = config.feature_dim

        data_scp = getattr(config, "%sfile" % partition)
        labels, data = zip(*read_mat_scp(data_scp))

        words = [re.split("_", x)[0] for x in labels]
        uwords = np.unique(words)

        word2id = {v: k for k, v in enumerate(uwords)}
        ids = [word2id[w] for w in words]
        # ids contain word to id mapping, ids size = num examples
        if feature_mean is None:
            feature_mean, n = 0.0, 0
            for x in data:
                feature_mean += np.sum(x)
                n += np.prod(x.shape)
            feature_mean /= n
        self.feature_mean = feature_mean

        
        self.data = np.array([x - self.feature_mean for x in data], dtype=object)
        self.ids = np.array(ids, dtype=np.int32)
        self.id_counts = Counter(ids)
        # id_counts is how many times each word occur
        print(self.id_counts)
        self.num_classes = len(self.id_counts)
        self.num_examples = len(self.ids)
        logger = logging.getLogger()
        #logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.DEBUG)
        logging.error("System reported: %s", self.ids)
        print(uwords)

    def shuffle(self):
        """Shuffle data."""

        shuffled_indices = np.random.permutation(self.num_examples)
        self.data = self.data[shuffled_indices]
        self.ids = self.ids[shuffled_indices]


    def pad_features(self, indices):
        """Pad acoustic features to max length sequence."""
        #indices contain all utteranceid in a mini batch
        #same utterances and different utterances 
        b = len(indices)
        lens = np.array([len(xx) for xx in self.data[indices]], dtype=np.int32)
        print("lens same ",lens)
        padded = np.zeros((b, max(lens), self.feature_dim))
        for i, (x, l) in enumerate(zip(self.data[indices], lens)):
            padded[i, :l] = x
        # padded is the features
        # lens is the list of length of the features
        return padded, lens, self.ids[indices]


    def batch(self, batch_size, max_same=1, max_diff=1):
        """Batch data."""
        # len(same) = 224 = 1*32 + max_same*32 + max_diff*32 = 32 + 32 + 160
        # len(diff) = 160 = max_diff*32
        self.shuffle()

        same = []
        for index, word_id in enumerate(self.ids):  # collect same samples
            # for each word find the same word
            # the size of same = num_examples 
            # for each index, it stores the index of same word
            # shape same  (10959, 1), shape diff  (10959, 5), type diff  <class 'numpy.ndarray'>
            # value diff  [[ 5793  1052   638  7449  3084]
            # [ 4564  3630 10646  4874  6940]
            indices = np.where(self.ids == word_id)[0]
            same.append(np.random.permutation(indices[indices != index])[:max_same])
            #print("len same ",len(same))
            #print("value same ",same[-1])
        same = np.array(same)
        #print("num examples",self.num_examples)
        print("value same ",same)
        #print("type same ",type(same))
        #print("len same ",len(same))
        #print("shape same ",same.shape)
        diff_ids = np.random.randint(0, self.num_classes - 1, (self.num_examples, max_diff))
        diff_ids[diff_ids >= np.tile(self.ids.reshape(-1, 1), [1, max_diff])] += 1

        diff = np.full_like(diff_ids, 0, dtype=np.int32)
        for word_id, count in self.id_counts.items():  # collect diff samples
            indices = np.where(diff_ids == word_id)
            diff[indices] = np.where(self.ids == word_id)[0][np.random.randint(0, count, len(indices[0]))]

        #print("value diff ",diff)
        #print("type diff ",type(diff))
        #print("len diff ",len(diff))
        #print("shape diff ",diff.shape)
        get_batch_indices = lambda start: range(start, min(start + batch_size, self.num_examples))

        for indices in map(get_batch_indices, range(0, self.num_examples, batch_size)):

            if self.is_train:
                b = len(indices)
                #print(indices) range(9760, 9792), 32 indices same as batchsize
                # np.arange(5) array([0, 1, 2, 3, 4])
                same_partition = [np.arange(b)]  # same segment ids for anchors
                same_partition += [(b + i) * np.ones(len(x)) for i, x in enumerate(same[indices])]  # same segment ids for same examples
                same_partition += [(2 * b) + np.arange(max_diff * b)]  # same segment ids for diff examples
                same_partition = np.concatenate(same_partition)

                diff_partition = np.concatenate([i * np.ones(max_diff) for i in range(b)])  # diff segment ids for diff examples
                print("len same ", len(same_partition))
                print("len same ", len(diff_partition))
                #cat1 = np.hstack(same[indices])   32
                #cat2 = diff[indices].flatten()    160
                #indices1 = np.concatenate((indices, cat1, cat2))   32 + 32 + 160
                indices = np.concatenate((indices, np.hstack(same[indices]), diff[indices].flatten()))
                # data is paded data   
                data, lens, _ = self.pad_features(indices)
                yield data, lens, same_partition, diff_partition

            else:
                yield self.pad_features(indices)
