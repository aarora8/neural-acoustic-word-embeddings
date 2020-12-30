from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from os import path, makedirs
import numpy as np
from model import Model
from data import Dataset

class Config(object):
    """Set up model for debugging."""

    trainfile = "../kaldi/data/len6-50frames-count2/train/mfcc.scp"
    devfile = "../kaldi/data/len6-50frames-count2/dev/mfcc.scp"
    batch_size = 32
    current_epoch = 0
    num_epochs = 100
    feature_dim = 39
    num_layers = 3
    hidden_size = 256
    bidirectional = True
    keep_prob = 0.7
    margin = 0.5
    max_same = 1
    max_diff = 5
    lr = 0.001
    mom = 0.9
    logdir = "../logs/test"
    ckptdir = "../ckpts/test"
    log_interval = 10
    ckpt = None
    debugmode = True

    makedirs(logdir, exist_ok=True)
    makedirs(ckptdir, exist_ok=True)


def main():
    config = Config()

    train_data = Dataset(partition="train", config=config)
    batch_size = config.batch_size
    total_batch=0
    for x, ts, same, diff in train_data.batch(batch_size, config.max_same, config.max_diff):
        total_batch=total_batch+1
        break
        #len same  224
        #len diff  160
        #print("len same ",len(same))
        #print("len diff ",len(diff))
    print(len(same))
    print(len(diff))
    print(x.shape)
    print(ts)
    
#    dev_data = Dataset(partition="dev", config=config, feature_mean=train_data.feature_mean)
#
#    train_model = Model(is_train=True, config=config, reuse=None)
#    dev_model = Model(is_train=False, config=config, reuse=True)
#
#    batch_size = config.batch_size
#
#    saver = tf.train.Saver()
#
#    proto = tf.ConfigProto(intra_op_parallelism_threads=2)
#    with tf.Session(config=proto) as sess:
#        sess.run(tf.global_variables_initializer())
#        sess.run(tf.local_variables_initializer())
#
#        ckpt = tf.train.get_checkpoint_state(config.ckptdir)
#        if ckpt and ckpt.model_checkpoint_path:
#            saver.restore(sess, ckpt.model_checkpoint_path)
#            print("restored from %s" % ckpt.model_checkpoint_path)
#
#        batch = 0
#        for epoch in range(config.current_epoch, config.num_epochs):
#            print("epoch: ", epoch)
#
#            losses = []
#            for x, ts, same, diff in train_data.batch(batch_size, config.max_same, config.max_diff):
#                _, loss = train_model.get_loss(sess, x, ts, same, diff)
#                losses.append(loss)
#                if batch % config.log_interval == 0:
#                    print("avg batch loss: %.4f" % np.mean(losses[-config.log_interval:]))
#                batch += 1
#
#            embeddings, labels = [], []
#            for x, ts, ids in dev_data.batch(batch_size):
#                embeddings.append(dev_model.get_embeddings(sess, x, ts))
#                labels.append(ids)
#            embeddings, labels = np.concatenate(embeddings), np.concatenate(labels)
#            print("ap: %.4f" % average_precision(embeddings, labels))
#
#            saver.save(sess, path.join(config.ckptdir, "model"), global_step=epoch)

if __name__ == "__main__":
    main()
