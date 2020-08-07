from __future__ import absolute_import, division, print_function, unicode_literals


from tensorflow import contrib
import matplotlib.pyplot as plt


import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from pandas import read_csv
tf.compat.v1.enable_eager_execution()

import csv
import os
import glob
import pandas as pd

pathway = '/Users/adamyuan/Projects/MachineLearning/VoiceGender/VoicePrint/TRAINCSV/'



def newcsvs(files):
    
    for f in files:
        if f.endswith("nm.csv"):
            continue
        if not f.endswith(".csv"):
            continue
        if f.startswith("new"):
            continue
        if f.startswith("merge"):
            continue
        
        inputfile = open(pathway + f, 'r')
        outputfile = open(pathway + "new" + f, 'w')
        data = list(inputfile)
        k = 0
        for row in data:
            if (k >= 100): break
               #skip header
            if (k == 0):
                k += 1
                continue
            result = [x.strip() for x in row.split(',')]
            k+=1
            for i in range(39):
                outputfile.write(str(result[i]) + ", ")
            if (f.startswith("Male")):
                outputfile.write("0")
            else:
                outputfile.write("1")
            outputfile.write("\n")
            


files1 = os.listdir(pathway)
newcsvs(files1)

def mergecsv(csvname, files):
    mergedfile = open(pathway + csvname + ".csv", "w")
    hearder = True
    for f in files:
        if f.endswith("nm.csv"):
            continue
        if not f.endswith(".csv"):
            continue
        if not f.startswith("new"):
            continue
        inputfile = open(pathway + f, 'r')

        data = list(inputfile)
        
        if hearder:
            hearder = False
            for i in range(13):
                mergedfile.write(str(i) + ", ")
            mergedfile.write("13")
            mergedfile.write("\n")

        for col in range(13):
            sum = 0
            s = len(data) % 8
            for row in data[s:-s]:
                result = [x.strip() for x in row.split(",")]
                sum += float(result[col])
            mergedfile.write(str(sum / len(data)) + ", ")
        result = [x.strip() for x in data[0].split(",")]
        mergedfile.write(result[-1])
        mergedfile.write("\n")

    mergedfile.close()
                    

mergecsv("merged", files1)

column_names = []
for i in range(14):
    column_names.append(str(i))

feature_names = column_names[:-1]
label_name = column_names[-1]
class_names = ['Male', 'Female']


def pack_features_vector(features, labels):
  """Pack the features into a single array."""
  features = tf.stack(list(features.values()), axis=1)
  return features, labels


batch_size = 32
train_dataset_fp1 = pathway + "merged.csv"
train_dataset = tf.data.experimental.make_csv_dataset(
                    train_dataset_fp1,
                    batch_size,
                    column_names=column_names,
                    label_name=label_name,
                    num_epochs=1, shuffle=True)

train_dataset = train_dataset.map(pack_features_vector)
features, labels = next(iter(train_dataset))


'''
def csvs_datasets(pathname, batch_size, column_names, label_name):
    features = []
    labels = []
    datasets = []
    files = os.listdir(pathname)
    for f in files:
        if f.endswith(".csv"):
            if f.startswith("new"):
                train_dataset_fp1 = pathname + f
                train_dataset = tf.data.experimental.make_csv_dataset(
                    train_dataset_fp1,
                    batch_size,
                    column_names=column_names,
                    label_name=label_name,
                    num_epochs=1)
                if f.startswith("newM"):
                    labels.append(0)
                else:
                    labels.append(1)
                train_dataset = train_dataset.map(pack_features_vector)
                feature, label = (next(iter(train_dataset)))
                features.append(feature)
                datasets.append(train_dataset)
    return (datasets, features, labels)

train_datasets, features, labels = csvs_datasets(pathway, 68, column_names, label_name)
features = tf.convert_to_tensor(features)
new_features = []
new_labels = []
for i in range(0, len(features), 2):
    new_features.append([features[i], features[i+1]])
    new_labels.append([labels[i], labels[i+1]])

features = tf.convert_to_tensor(new_features)
labels = tf.convert_to_tensor(new_labels)

print(features)

print(len(features), len(labels))
'''

model = tf.keras.Sequential([
    tf.keras.layers.Dense(39, activation=tf.nn.relu,input_shape=(13, )
                          ),  # input shape required
    tf.keras.layers.Dense(26, activation=tf.nn.relu),
    tf.keras.layers.Dense(13, activation=tf.nn.relu),
    tf.keras.layers.Dense(2, activation=tf.nn.sigmoid)
])


predictions = model(features)
print(predictions[:5])

tf.nn.softmax(predictions[:5])

print("Prediction: {}".format(tf.argmax(predictions, axis=1)))
print("    Labels: {}".format(labels))


def loss(model, x, y):
  y_ = model(x)
  return tf.compat.v1.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)


l = loss(model, features, labels)
print("Loss test: {}".format(l))


def grad(model, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value = loss(model, inputs, targets)
  return loss_value, tape.gradient(loss_value, model.trainable_variables)


optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01)

global_step = tf.Variable(0)

loss_value, grads = grad(model, features, labels)

print("Step: {}, Initial Loss: {}".format(global_step.numpy(),
                                          loss_value.numpy()))

optimizer.apply_gradients(zip(grads, model.trainable_variables), global_step)

print("Step: {},         Loss: {}".format(global_step.numpy(),
                                          loss(model, features, labels).numpy()))

tfe = contrib.eager

# keep results for plotting
train_loss_results = []
train_accuracy_results = []

num_epochs = 201

for epoch in range(num_epochs):
  epoch_loss_avg = tfe.metrics.Mean()
  epoch_accuracy = tfe.metrics.Accuracy()

  # Training loop - using batches of 32
  for x, y in train_dataset:
    # Optimize the model
    loss_value, grads = grad(model, x, y)
    optimizer.apply_gradients(zip(grads, model.trainable_variables),
                              global_step)

    # Track progress
    epoch_loss_avg(loss_value)  # add current batch loss
    # compare predicted label to actual label
    epoch_accuracy(tf.argmax(model(x), axis=1, output_type=tf.int32), y)

  # end epoch
  train_loss_results.append(epoch_loss_avg.result())
  train_accuracy_results.append(epoch_accuracy.result())

  if epoch % 20 == 0:
    print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                epoch_loss_avg.result(),
                                                                epoch_accuracy.result()))


pathway = '/Users/adamyuan/Projects/MachineLearning/VoiceGender/VoicePrint/TESTCSV/'

files2 = os.listdir(pathway)

newcsvs(files2)
mergecsv("mergedtest", files2)

test_fp = pathway + "mergedtest.csv"

test_dataset = tf.data.experimental.make_csv_dataset(
    test_fp,
    batch_size,
    column_names=column_names,
    label_name='13',
    num_epochs=1,
    shuffle=True)

test_dataset = test_dataset.map(pack_features_vector)

test_accuracy = tfe.metrics.Accuracy()

for (x, y) in test_dataset:
  logits = model(x)
  prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
  test_accuracy(prediction, y)

print("Test set accuracy: {:.3%}".format(test_accuracy.result()))


print(tf.stack([y, prediction], axis=1))

'''prediction
predict_dataset = tf.convert_to_tensor([
    [5.1, 3.3, 1.7, 0.5,],
    [5.9, 3.0, 4.2, 1.5,],
    [6.9, 3.1, 5.4, 2.1]
])

predictions = model(predict_dataset)

for i, logits in enumerate(predictions):
  class_idx = tf.argmax(logits).numpy()
  p = tf.nn.softmax(logits)[class_idx]
  name = class_names[class_idx]
  print("Example {} prediction: {} ({:4.1f}%)".format(i, name, 100*p))
'''


