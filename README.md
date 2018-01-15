# CNN CIFAR-10

A 5-layer CNN (4 convolutional layers + 1 fully connected) is built. Its performances are tested in terms of accuracy on CIFAR-10 dataset (converted in TFRecords format). The model is built using the new TensorFlow API (tf.estimator).

### STEP ONE: Covert CIFAR-10 in TFRecords
To achieve this task, a script from TensorFlow Tutorial was used (Is a standard script that everybody uses)
```
$: python generate_cifar10_tfrecords.py
```
The same, can be done using a similar script from TFSlim Library.

### STEP TWO: Train and Evaluate the model
```
$: python cifar10_5cnn.py
```

Some details about the training:

* batch_size = 256
* Images are in channels_last format (better for CPU)
* Every layer has ReLu activation function
* learning_rate = 0.0007 exponentially decaying every 10 epochs.
* Unfortunately the training was done on my laptop CPU (2,5 GHz Intel Core i5)

### STEP THREE: Visualize results on TensorBoard
```
$ tensorboard --logdir=./folder_where_checkpoints_are_stored
```
Then connect your browser on $0.0.0.0:6006$
