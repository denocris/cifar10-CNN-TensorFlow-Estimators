""" Exercise: 5 layer CNN on Cifar-10 (TFRecords) using tf.estimator API

    To run the code:
    $: python cifar10_5cnn.py

    Results can be seen on Tensorboard:
    $: tensorboard --logdir=./folder_where_checkpoints_are_stored
"""

from argparse import ArgumentParser
import os
import tensorflow as tf


def cnn_model(features, mode, params):

    is_training = mode == tf.estimator.ModeKeys.TRAIN

    #print("------------- before ----------------", features.get_shape())
    with tf.name_scope('Input'):
        # Input Layer
        input_layer = tf.reshape(features, [-1, 32, 32, 3], name='input_reshape')
        tf.summary.image('input', input_layer)
    #print("------------- after -----------------", input_layer.get_shape())

    with tf.name_scope('Conv_1'):
        # Convolutional Layer #1
        conv1 = tf.layers.conv2d(
          inputs=input_layer,
          filters=32,
          kernel_size=(5, 5),
          padding='same',
          activation=tf.nn.relu,
          trainable=is_training,
          data_format='channels_last')
        tf.summary.histogram('Convolution_layers/conv1', conv1)

        # Pooling Layer #1
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=(2, 2), strides=2, padding='same')
        norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

        if params.print_shape:
            # to check the expected shape
            print("------- Conv_1 ----------", pool1.get_shape())

    with tf.name_scope('Conv_2'):
        # Convolutional Layer #2 and Pooling Layer #2
        conv2 = tf.layers.conv2d(
            inputs=norm1,
            filters=64,
            kernel_size=(5, 5),
            padding='same',
            activation=tf.nn.relu,
            trainable=is_training,
            data_format='channels_last')
        tf.summary.histogram('Convolution_layers/conv2', conv2)

        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=(2, 2), strides=2, padding='same')
        norm2 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')

        if params.print_shape:
            # to check the expected shape
            print("------- Conv_2 ----------", pool2.get_shape())

    with tf.name_scope('Conv_3'):
        # Convolutional Layer #3 and Pooling Layer #3
        conv3 = tf.layers.conv2d(
            inputs=norm2,
            filters=96,
            kernel_size=(3, 3),
            padding='same',
            activation=tf.nn.relu,
            trainable=is_training,
            data_format='channels_last')
        tf.summary.histogram('Convolution_layers/conv3', conv3)

        pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=(2, 2), strides=2, padding='same')
        norm3 = tf.nn.lrn(pool3, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm3')

        if params.print_shape:
            # to check the expected shape
            print("------- Conv_3 ----------", pool3.get_shape())

    with tf.name_scope('Conv_4'):
        # Convolutional Layer #4 and Pooling Layer #4
        conv4 = tf.layers.conv2d(
            inputs=norm3,
            filters=64,
            kernel_size=(3, 3),
            padding='same',
            activation=tf.nn.relu,
            trainable=is_training,
            data_format='channels_last')
        tf.summary.histogram('Convolution_layers/conv4', conv4)

        norm4 = tf.nn.lrn(conv4, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm4')
        pool4 = tf.layers.max_pooling2d(inputs=norm4, pool_size=(2, 2), strides=1, padding='same')

        if params.print_shape:
            # to check the expected shape
            print("------- Conv_4 ----------", pool4.get_shape())


    with tf.name_scope('Dense_Dropout'):
        # Dense Layer
        pool_flat = tf.contrib.layers.flatten(pool4)
        dense = tf.layers.dense(inputs=pool_flat, units=1024, activation=tf.nn.relu, trainable=is_training)
        dropout = tf.layers.dropout(inputs=dense, rate=params.dropout_rate, training=is_training)
        tf.summary.histogram('fully_connected_layers/dropout', dropout)


    with tf.name_scope('Predictions'):
        # Logits Layer
        logits = tf.layers.dense(inputs=dropout, units=10, trainable=is_training)
        return logits

def cnn_model_fn(features, labels, mode, params):
    """Model function for CNN."""

    logits = cnn_model(features, mode, params)
    predicted_logit = tf.argmax(input=logits, axis=1, output_type=tf.int32)
    scores = tf.nn.softmax(logits, name='softmax_tensor')

    # Generate Predictions
    predictions = {
      'classes': predicted_logit,
      'probabilities': scores
    }

    export_outputs = {
        'prediction': tf.estimator.export.ClassificationOutput(
            scores=scores,
            classes=tf.cast(predicted_logit, tf.string))
    }

    # For PREDICTION mode
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, export_outputs=export_outputs)

    # For TRAIN and EVAL modes
    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)

    accuracy = tf.metrics.accuracy(tf.argmax(labels, axis=1), predicted_logit)
    train_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(labels, axis=1, output_type=tf.int32), predicted_logit), tf.float32))

    eval_metric = { 'test_accuracy': accuracy }

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        #tf.summary.scalar('accuracy', accuracy[0])
        tf.summary.scalar('train_accuracy', train_accuracy)
        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.train.get_global_step(),
            learning_rate=params.learning_rate,
            learning_rate_decay_fn=lambda lr, step: tf.train.exponential_decay(params.learning_rate, tf.train.get_global_step(), 780, 0.94, staircase=True),
            optimizer='Adam')
    else:
        train_op = None

    # EstimatorSpec fully defines the model to be run by an Estimator.
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric, # A dict of name/value pairs specifying the metrics that will be calculated when the model runs in EVAL mode.
        predictions=predictions,
        export_outputs=export_outputs)

def data_input_fn(filenames, batch_size=1000, shuffle=False):

    def _parser(serialized_example):

        features = tf.parse_single_example(
            serialized_example,
            features={
                'image': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.int64),
            })
        image = tf.decode_raw(features['image'], tf.uint8)
        image.set_shape([3 * 32 * 32])

        # Reshape from [depth * height * width] to [depth, height, width].
        image = tf.cast(
            tf.transpose(tf.reshape(image, [3, 32, 32]), [1, 2, 0]),
            tf.float32)
        label = tf.cast(features['label'], tf.int32)

        return image, tf.one_hot(label, depth=10)

    def _input_fn():
        dataset = tf.data.TFRecordDataset(filenames)
        dataset = dataset.map(_parser)
        if shuffle:
            dataset = dataset.shuffle(buffer_size=10000)

        dataset = dataset.repeat(None)
        dataset = dataset.batch(batch_size)

        iterator = dataset.make_one_shot_iterator()
        features, labels = iterator.get_next()
        return features, labels
    return _input_fn

if __name__ == '__main__':

    tf.logging.set_verbosity(tf.logging.INFO)


    parser = ArgumentParser()
    parser.add_argument(
        "--data-directory",
        default='/Users/Yak52/my_datasets/tfrecords-cifar10-v4',
        help='Directory where TFRecord images are stored'
    )
    parser.add_argument(
        '--model-directory',
        default='/Users/Yak52/Github/cifar10_cnn/checks-cifar-latest',
        help='Directory where model summaries and checkpoints are stored'
    )
    args = parser.parse_args()

    run_config = tf.contrib.learn.RunConfig(
        model_dir=args.model_directory,
        save_checkpoints_steps=20,
        save_summary_steps=20)

    hparams = tf.contrib.training.HParams(
        learning_rate=0.0007,
        dropout_rate=0.4,
        data_directory=os.path.expanduser(args.data_directory),
        print_shape=False)


    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn,
        config=run_config,
        params=hparams
    )


    train_batch_size = 256 #256
    eval_batch_size = 64 #128
    train_steps = 40000 // train_batch_size # len dataset // batch size

    train_input_fn = data_input_fn(os.path.join(hparams.data_directory, 'cifar10_train.tfrecords'), batch_size=train_batch_size)
    eval_input_fn = data_input_fn(os.path.join(hparams.data_directory, 'cifar10_eval.tfrecords'), batch_size=eval_batch_size)

    experiment = tf.contrib.learn.Experiment(
        mnist_classifier,
        train_input_fn=train_input_fn,
        eval_input_fn=eval_input_fn,
        train_steps=train_steps * 30,
        eval_steps=100,
        min_eval_frequency=1
    )

    experiment.train_and_evaluate()

    # Export for serving
    # mnist_classifier.export_savedmodel(
    #     os.path.join(hparams.data_directory, 'serving'),
    #     serving_input_receiver_fn
    # )
