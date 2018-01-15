# CNN CIFAR-10
In *cifar10_5cnn.py*, a 5-layer CNN (4 convolutional layers + 1 fully connected) is built. The model is coded using the new TensorFlow API (tf.estimator).
Its performances are tested in terms of accuracy on CIFAR-10 dataset (converted in TFRecords format).

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

Unfortunately, the training was done on my laptop CPU (2,5 GHz Intel Core i5). At the moment I do not have any GPU at my disposal.

Some details about the training:

* Training dataset size = 40000, test dataset size = 10000;
* Images are in channels_last format (better for CPU);
* Every layer has a ReLu activation function;
* learning_rate = 0.007 exponentially decaying every 10 epochs, batch_size = 256;
* Loss function is the Softmax-Cross-Entropy;
* dropout_rate = 0.4 (only in TRAIN mode);

In model_1, the four convolutional layers have [32,64,96,64] respectively. Its fully connected layer has units=1024.

In model_2, the four convolutional layers have [32,64,64,32] respectively. Its fully connected layer has units=512. This model was built in order to test a less complex model (less parameters).


### STEP THREE: Visualize results on TensorBoard
```
$ tensorboard --logdir=./checks-cifar-latest
```
Then connect your browser on *0.0.0.0:6006*.

### RESULTS
The following plots were generated using plots.ipynb.
![Accuracy](/plots/accuracy.png)
Train accuracy is higher than test accuracy. This is a sign of overfitting. Unfortunately, I did not have enough computational time to better analyze the landscape of hyperparameters in order to reduce overfitting. Model_2 was just an attempt to reduce complexity (reducing the number of conv-filters).

![Loss](/plots/loss.png)
The loss function is not yet stabilized, so the training could have go on ...
