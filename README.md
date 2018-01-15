# CNN CIFAR-10

A 5-layer CNN (4 convolutional layers + 1 fully connected) is built. Its performances are tested in terms of accuracy on CIFAR-10 dataset (converted in TFRecords format). The model is coded using the new TensorFlow API (tf.estimator).

### STEP ONE: Covert CIFAR-10 in TFRecords
To achieve this task, a script from TensorFlow Tutorial was used (is a standard script, often used by the community)
```
$: python generate_cifar10_tfrecords.py
```
Another standard script to convert data in TFRecords can be found in TFSlim Library. Before running the script, different classes must be organized in different folders.

### STEP TWO: Train and Evaluate the model
```
$: python cifar10_5cnn.py
```

Unfortunately the training was done on my laptop CPU (2,5 GHz Intel Core i5). At the moment I do not have GPUs at my disposal.

Some details about the training:

* Training dataset size = 40000, test dataset size = 10000;
* Images are in channels_last format (better for CPU);
* Every layer has ReLu activation function;
* learning_rate = 0.0007 exponentially decaying every 10 epochs, batch_size = 256;
* Loss function is the Softmax-Cross-Entropy;
* dropout_rate = 0.4 (only in TRAIN mode);


### STEP THREE: Visualize results on TensorBoard
```
$ tensorboard --logdir=./folder_where_checkpoints_are_stored
```
Then connect your browser on $0.0.0.0:6006$

### RESULTS
The following plots were generated using plots.ipynb.
![Accuracy](/plots/accuracy.png)
Train accuracy is higher than test accuracy. This is a sign of overfitting. Unfortunately I did not have enough computational time to better analyze the landscape of hyperparameters in order to reduce overfitting. Model_2 was an attempt to reduce complexity (reducing the number of conv-filters).

![Loss](/plots/loss.png)
