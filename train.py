import os
import argparse
import datetime
import numpy as np
import torch
from progressbar import *

from src.params import Params
from src.model import face_model
from src.data   import get_dataloader
from src.triplet_loss import batch_all_triplet_loss, batch_hard_triplet_loss, adapted_triplet_loss

import matplotlib.pyplot as plt

DISPLAY = False

class Trainer():
    
    def __init__(self, json_path, data_dir, valid_dir, validate, ckpt_dir, log_dir, restore):
        print("Reached __init__")

        self.params      = Params(json_path)
        self.valid       = validate
        self.model       = face_model(self.params)

        initial_learning_rate = self.params.learning_rate
        self.optimizer   = torch.optim.Adam(self.model.parameters(), lr=initial_learning_rate, betas=(0.9, 0.999), eps=0.1)
        self.lr_scheduler   = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=0.96)
        self.dictionary_embeddings = {}

        if self.params.triplet_strategy == "batch_all":
            self.loss = batch_all_triplet_loss
        elif self.params.triplet_strategy == "batch_hard":
            self.loss = batch_hard_triplet_loss
        elif self.params.triplet_strategy == "batch_adaptive":
            self.loss = adapted_triplet_loss

        # if restore:
#        self.checkpoint.restore(self.ckptmanager.latest_checkpoint)
        # print(f'\nRestored from Checkpoint : {self.ckptmanager.latest_checkpoint}\n')
        
        # else:
            # print('\nIntializing from scratch\n')
        
        print("Setting up train dataloader")
        self.train_dataloader = get_dataloader(data_dir, self.params, 'train')
        self.train_samples = len(self.train_dataloader)

        if self.valid:
            self.valid_dataloader = get_dataloader(valid_dir, self.params, 'val')
            self.valid_samples = len(self.valid_dataloader)
        
    def __call__(self, epoch):
        for i in range(epoch):
            self.train(i)
            if self.valid:
                with torch.no_grad():
                    self.validate(i)

        
    def train(self, epoch, last=False):
        # widgets = [f'Train epoch {epoch} :', Percentage(), ' ', Bar('#'), ' ',Timer(), ' ', ETA(), ' ']
        # pbar = ProgressBar(widgets=widgets, max_value=int(self.train_samples // self.params.batch_size) + 20).start()
        total_loss = 0

        for i, (images, labels) in enumerate(self.train_dataloader):
            loss = self.train_step(images, labels, last)
            total_loss += loss
            
            print('train_step_loss: {}'.format(loss))
            print('train_batch_loss: {}'.format(total_loss))
        
        if (epoch + 1) % 5 == 0:
            print('\nTrain Loss over epoch {}: {}'.format(epoch, total_loss))
            save_path = 'model.pt'
            torch.save(self.model.state_dict(), save_path)
            print(f'Saved Checkpoint for step {epoch+1} : {save_path}\n')

        if last:
            np.save('embeddings.npy', self.dictionary_embeddings)
            
    def validate(self, epoch):
        # widgets = [f'Valid epoch {epoch} :', Percentage(), ' ', Bar('#'), ' ',Timer(), ' ', ETA(), ' ']
        # pbar = ProgressBar(widgets=widgets, max_value=int(self.valid_samples // self.params.batch_size) + 50).start()
        total_loss = 0

        for i, (images, labels) in enumerate(self.valid_dataset):
            loss = self.valid_step(images, labels)
            total_loss += loss
            print("valid_step_loss: {}".format(loss))
            print("valid_batch_loss: {}".format(total_loss))
        
        if (epoch+1)%5 == 0:
            print('\nValidation Loss over epoch {}: {}\n'.format(epoch, total_loss)) 
    
    step_count = 0
    def train_step(self, images, labels, last):
        self.optimizer.zero_grad()

        embeddings = self.model(images)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1, eps=1e-10)

        #print(len(images), labels, last)

        loss = self.loss(labels, embeddings, self.params.margin, self.params.squared)

        if last:
            embeddings_ = embeddings.detach().cpu().numpy()
            batch_size = embeddings_.shape[0]
            for i in range(batch_size):
                if DISPLAY:
                    plt.title(labels.numpy()[i].decode('utf-8'))
                    plt.ylim((0,40))

                    plt.hist(embeddings_[i, :])
                    plt.show()

                label = self.train_dataloader.image_labels_unique[labels.detach().cpu().numpy()[i]]

                # Check if the dictionary has the label stored.
                if label not in self.dictionary_embeddings:
                    self.dictionary_embeddings[label] = embeddings_[i,:]
                else:
                    self.dictionary_embeddings[label] = np.add(embeddings_[i,:], self.dictionary_embeddings[label])
                    self.dictionary_embeddings[label] = self.dictionary_embeddings[label] / 2


        loss.backward()
        self.step_count += 1
        if self.step_count % 10_000 != 0:
            self.optimizer.step()
        else:
            self.lr_scheduler.step()
        return loss
    
    
    def valid_step(self, images, labels):
        embeddings = self.model(images)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1, eps=1e-10)
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
    
    print("Setting up trainer")
    trainer = Trainer(args.params_dir, args.data_dir, args.validate, 1, args.ckpt_dir, args.log_dir, args.restore)

    print("Training")    
    for i in range(args.epoch):
        if i is (args.epoch - 1):
            trainer.train(i, True)
        else:
            trainer.train(i)
        # trainer.validate(i)


# 1 record - /root/shared_folder/Harish/Facenet/data
# 10 records - /root/shared_folder/Amaan/face/FaceNet-and-FaceLoss-collections-tensorflow2.0/data10faces_aligned_tfrcd
# Complete record - /root/shared_folder/Amaan/face/FaceNet-and-FaceLoss-collections-tensorflow2.0/data2/