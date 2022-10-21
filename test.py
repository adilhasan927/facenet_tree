import os
import argparse
import cv2
import tensorflow as tf
import numpy as np

from src.params import Params
from src.model  import face_model
from src.triplet_loss import adapted_triplet_loss
from sklearn.metrics import confusion_matrix, classification_report

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
image_size = 250

class Tester():

    def __init__(self, json_path, test_dir, ckpt_dir, log_dir, dictionary_embeddings):

        self.params = Params(json_path)
        self.model = face_model(self.params)
        self.dictionary_embeddings = np.load(dictionary_embeddings,allow_pickle='TRUE').item()
        self.test_dir = test_dir

        self.checkpoint = tf.train.Checkpoint(model=self.model,
                                              train_steps=tf.Variable(0, dtype=tf.int64),
                                              valid_steps=tf.Variable(0, dtype=tf.int64),
                                              epoch=tf.Variable(0, dtype=tf.int64))
        self.ckptmanager = tf.train.CheckpointManager(self.checkpoint, ckpt_dir, 3)

        self.loss = adapted_triplet_loss
        self.train_summary_writer = tf.summary.create_file_writer(log_dir)

        self.checkpoint.restore(self.ckptmanager.latest_checkpoint)
        print(f'\nRestored from Checkpoint : {self.ckptmanager.latest_checkpoint}\n')

    def __call__(self, epoch):
        self.test()

    def calculate_l2_norm(self, embedding, reference):
        return np.linalg.norm(embedding - reference)

    def test(self):
        # widgets = [f'Valid epoch {epoch} :', Percentage(), ' ', Bar('#'), ' ',Timer(), ' ', ETA(), ' ']
        # pbar = ProgressBar(widgets=widgets, max_value=int(self.valid_samples // self.params.batch_size) + 50).start()
        total_loss = 0
        test_result = []
        test_labels = []


        for file in os.listdir(self.test_dir):
            # Store the labels
            label = file.split('.')[0]
            label = ''.join(filter(lambda x: not x.isdigit(), label)).lower()
            test_labels.append(label)

            # Store the inference
            feature = np.load(self.test_dir + file, allow_pickle=True)
            # select_idx = [0, 1, 2, 3, 4, 20, 21, 22, 23, 24]
            # feature = feature[:,:,select_idx]
            res = cv2.resize(feature, dsize=(image_size, image_size), interpolation=cv2.INTER_CUBIC)
            test_result.append(self.test_step(res))

        # Print out the confusion matrix
        print(confusion_matrix(test_labels, test_result))
        print()
        print(classification_report(test_labels, test_result))

    def test_step(self, image):
        image = tf.expand_dims(image, axis=0)
        embedding = self.model(image)
        embedding = tf.math.l2_normalize(embedding, axis=1, epsilon=1e-10)
        calculated_distances = {}

        # Iteratively go through the dictionary of embeddings and find the minimum distance
        for key in self.dictionary_embeddings.keys():
            distance = self.calculate_l2_norm(embedding.numpy(), self.dictionary_embeddings[key])
            calculated_distances[key] = distance

        # Return the minimum distance, that should be the most suitable class
        return min(calculated_distances, key=calculated_distances.get)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--params_dir', default='hyperparameters/batch_adaptive.json',
                        help="Experiment directory containing params.json")
    parser.add_argument('--test_dir', default='test/',
                        help="Directory containing testing data")
    parser.add_argument('--ckpt_dir', default='.tf_ckpt/',
                        help="Directory containing the Checkpoints")
    parser.add_argument('--log_dir', default='.logs/',
                        help="Directory containing the Logs")
    parser.add_argument('--dictionary_embeddings', default='embeddings.npy',
                        help="Dictionary containing a whole repository of embeddings")
    args = parser.parse_args()

    tester = Tester(args.params_dir, args.test_dir, args.ckpt_dir, args.log_dir, args.dictionary_embeddings)
    tester.test()