import os
import argparse
import datetime
import numpy as np
import tensorflow as tf
from progressbar import *

from src.params import Params
from src.model import face_model
from src.data   import get_dataset
from src.triplet_loss import batch_all_triplet_loss, batch_hard_triplet_loss, adapted_triplet_loss

import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
DISPLAY = False

class Trainer():
    
    def __init__(self, json_path, data_dir, valid_dir, validate, ckpt_dir, log_dir, restore):
        
        self.params      = Params(json_path)
        self.valid       = validate
        self.model       = face_model(self.params)

        self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(self.params.learning_rate,
                                                                          decay_steps=10000, decay_rate=0.96, staircase=True)
        self.optimizer   = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule, beta_1=0.9, beta_2=0.999, epsilon=0.1)
        self.checkpoint  = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer, train_steps=tf.Variable(0,dtype=tf.int64),
                                               valid_steps=tf.Variable(0,dtype=tf.int64), epoch=tf.Variable(0, dtype=tf.int64))
        self.ckptmanager = tf.train.CheckpointManager(self.checkpoint, ckpt_dir, 3)
        self.dictionary_embeddings = {}

        if self.params.triplet_strategy == "batch_all":
            self.loss = batch_all_triplet_loss
            
        elif self.params.triplet_strategy == "batch_hard":
            self.loss = batch_hard_triplet_loss
            
        elif self.params.triplet_strategy == "batch_adaptive":
            self.loss = adapted_triplet_loss
            
        current_time = datetime.datetime.now().strftime("%d-%m-%Y_%H%M%S")
        log_dir += current_time + '/train/'
        self.train_summary_writer = tf.summary.create_file_writer(log_dir)
            
        # if restore:
        self.checkpoint.restore(self.ckptmanager.latest_checkpoint)
        # print(f'\nRestored from Checkpoint : {self.ckptmanager.latest_checkpoint}\n')
        
        # else:
            # print('\nIntializing from scratch\n')
            
        self.train_dataset, self.train_samples = get_dataset(data_dir, self.params, 'train')
        
        if self.valid:
            self.valid_dataset, self.valid_samples = get_dataset(valid_dir, self.params, 'val')
        
        
    def __call__(self, epoch):
        
        for i in range(epoch):
            self.train(i)
            if self.valid:
                self.validate(i)

        
    def train(self, epoch, last=False):
        # widgets = [f'Train epoch {epoch} :', Percentage(), ' ', Bar('#'), ' ',Timer(), ' ', ETA(), ' ']
        # pbar = ProgressBar(widgets=widgets, max_value=int(self.train_samples // self.params.batch_size) + 20).start()
        total_loss = 0

        for i, (images, labels) in enumerate(self.train_dataset):
            loss = self.train_step(images, labels, last)
            total_loss += loss
            
            with self.train_summary_writer.as_default():
                tf.summary.scalar('train_step_loss', loss, step=self.checkpoint.train_steps)
            self.checkpoint.train_steps.assign_add(1)

        if last:
            # Dump the characteristic embeddings into a pickle
            np.save('embeddings.npy', self.dictionary_embeddings)

        with self.train_summary_writer.as_default():
            tf.summary.scalar('train_batch_loss', total_loss, step=epoch)
        
        self.checkpoint.epoch.assign_add(1)
        if int(self.checkpoint.epoch) % 5 == 0:
            save_path = self.ckptmanager.save()
            print('\nTrain Loss over epoch {}: {}'.format(epoch, total_loss))
            print(f'Saved Checkpoint for step {self.checkpoint.epoch.numpy()} : {save_path}\n')

            
    def validate(self, epoch):
        # widgets = [f'Valid epoch {epoch} :', Percentage(), ' ', Bar('#'), ' ',Timer(), ' ', ETA(), ' ']
        # pbar = ProgressBar(widgets=widgets, max_value=int(self.valid_samples // self.params.batch_size) + 50).start()
        total_loss = 0

        for i, (images, labels) in enumerate(self.valid_dataset):
            loss = self.valid_step(images, labels)
            total_loss += loss
            
            with self.train_summary_writer.as_default():
                tf.summary.scalar('valid_step_loss', loss, step=self.checkpoint.valid_steps)
            self.checkpoint.valid_steps.assign_add(1)
        print('\n')
        with self.train_summary_writer.as_default():
            tf.summary.scalar('valid_batch_loss', total_loss, step=epoch)
        
        if (epoch+1)%5 == 0:
            print('\nValidation Loss over epoch {}: {}\n'.format(epoch, total_loss)) 
        
    def train_step(self, images, labels, last):

        with tf.GradientTape() as tape:
            embeddings = self.model(images)
            embeddings = tf.math.l2_normalize(embeddings, axis=1, epsilon=1e-10)
            loss = self.loss(labels, embeddings, self.params.margin, self.params.squared)
            if last:
                embeddings_ = embeddings.numpy()
                batch_size = embeddings_.shape[0]
                for i in range(batch_size):
                    if DISPLAY:
                        plt.title(labels.numpy()[i].decode('utf-8'))
                        plt.ylim((0,40))

                        plt.hist(embeddings_[i, :])
                        plt.show()

                    label = labels.numpy()[i].decode('utf-8')

                    # Check if the dictionary has the label stored.
                    if label not in self.dictionary_embeddings:
                        self.dictionary_embeddings[label] = embeddings_[i,:]
                    else:
                        self.dictionary_embeddings[label] = np.add(embeddings_[i,:], self.dictionary_embeddings[label])
                        self.dictionary_embeddings[label] = self.dictionary_embeddings[label] / 2


        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        
        return loss
    
    
    def valid_step(self, images, labels):
        
        embeddings = self.model(images)
        embeddings = tf.math.l2_normalize(embeddings, axis=1, epsilon=1e-10)
        loss = self.loss(labels, embeddings, self.params.margin, self.params.squared)
            
        return loss


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', default=10, type=int,
                        help="Number epochs to train the model for")
    parser.add_argument('--params_dir', default='hyperparameters/batch_adaptive.json',
                        help="Experiment directory containing params.json")
    parser.add_argument('--data_dir', default='../face-data/',
                        help="Directory containing the dataset")
    parser.add_argument('--validate', default='./val/',
                        help="Is there an validation dataset available")
    parser.add_argument('--ckpt_dir', default='.tf_ckpt/',
                        help="Directory containing the Checkpoints")
    parser.add_argument('--log_dir', default='.logs/',
                        help="Directory containing the Logs")
    parser.add_argument('--restore', default=0, type=int,
                        help="Restart the model from the previous Checkpoint")
    args = parser.parse_args()
    
    trainer = Trainer(args.params_dir, args.data_dir, args.validate, 1, args.ckpt_dir, args.log_dir, args.restore)
    
    for i in range(args.epoch):
        if i is (args.epoch - 1):
            trainer.train(i, True)
        else:
            trainer.train(i)
        # trainer.validate(i)


# 1 record - /root/shared_folder/Harish/Facenet/data
# 10 records - /root/shared_folder/Amaan/face/FaceNet-and-FaceLoss-collections-tensorflow2.0/data10faces_aligned_tfrcd
# Complete record - /root/shared_folder/Amaan/face/FaceNet-and-FaceLoss-collections-tensorflow2.0/data2/