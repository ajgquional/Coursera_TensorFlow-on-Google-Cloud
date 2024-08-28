# Coursera: TensorFlow on Google Cloud (by Google Cloud)

## About the repo
This repository contains Jupyter Notebooks, along with data files used, and SQL queries (all downloaded from Google Cloud) in the Coursera course "TensorFlow on Google Cloud" by Google Cloud. This repository is mainly for reference.

## About the directories
* <b>```Module3```</b> contains the labs about refactoring an existing linear regression model so that is takes its data from a ```tf.data.Dataset```, as well as using Keras to define the model, and preprocessing layers as a bridge to map from columns in a CSV to features used to train the model
* <b>```Module4```</b> contains labs about building a simple deep neural network model using the Keras Sequential API and Feature Columns then deploying it using Google Cloud AI Platform and see how to call the model for online prediciton; as well as building a Keras DNN to predict the fare amount for NYC taxi cab rides
* <b>```Module5```</b> contains a lab about training at scale with the Vertex AI training service

## Reading/Video links
The following table contains reading/video links provided in each module of the course (copied here for future reference).

<table>
  <tr>
    <th>Module Number</th>
    <th>Module Title</th>
    <th>Reading/Video link</th>
  </tr>

  <!-- To disable zebra striping-->
  <tr></tr>
  
  <!--Module 2 links-->
  <tr>
    <td rowspan=24 align="center">2</td>
    <td rowspan=24>Introduction to the TensorFlow Ecosystem</td>
    <td><a href="https://towardsdatascience.com/introduction-on-tensorflow-2-0-bd99eebcdad5">Introduction on TensorFlow 2.0</a></td>
  </tr>
  <tr></tr> <!-- To disable zebra striping-->
  <tr>
    <td><a href="https://towardsdatascience.com/a-quick-introduction-to-tensorflow-2-0-for-deep-learning-e740ca2e974c">Getting started with TensorFlow 2</a></td>
  </tr>
  <tr></tr> <!-- To disable zebra striping-->
  <tr>
    <td><a href="https://www.youtube.com/watch?v=zL3jjTtHklM&feature=youtu.be">ASL Webinar: TensorFlow with Ryan Giliard</a></td>
  </tr>
  <tr></tr> <!-- To disable zebra striping-->
  <tr>
    <td><a href="https://www.youtube.com/watch?v=5ECD8J3dvDQ">Introduction to TensorFlow 2.0: Easier for beginners, and more powerful for experts (TF World '19)</a></td>
  </tr>
  <tr></tr> <!-- To disable zebra striping-->
  <tr>
    <td><a href="https://www.youtube.com/watch?v=VwVg9jCtqaU">Machine Learning - Zero to Hero</a></td>
  </tr>
  <tr></tr> <!-- To disable zebra striping-->
  <tr>
    <td><a href="https://medium.com/ml-book/demonstration-of-tensorflow-feature-columns-tf-feature-column-3bfcca4ca5c4">Demonstration of TensorFlow Feature Columns (tf.feature_column)</a></td>
  </tr>
  <tr></tr> <!-- To disable zebra striping-->
  <tr>
    <td><a href="https://www.tensorflow.org/guide/tensor">Introduction to Tensors</a></td>
  </tr>
  <tr></tr> <!-- To disable zebra striping-->
  <tr>
    <td><a href="https://medium.com/@jinturkarmugdha/introduction-to-tensors-and-its-types-fc19da29bc56">Introduction to Tensors and its Types</a></td>
  </tr>
  <tr></tr> <!-- To disable zebra striping-->
  <tr>
    <td><a href="https://medium.com/mostly-ai/tensorflow-records-what-they-are-and-how-to-use-them-c46bc4bbb564">Tensorflow Records? What they are and how to use them</a></td>
  </tr>
  <tr></tr> <!-- To disable zebra striping-->
  <tr>
    <td><a href="https://www.tensorflow.org/tutorials/load_data/tfrecord">TFRecord and tf.train.Example</a></td>
  </tr>
  <tr></tr> <!-- To disable zebra striping-->
  <tr>
    <td><a href="https://towardsdatascience.com/hands-on-tensorflow-data-validation-61e552f123d7">Hands on Tensorflow Data Validation</a></td>
  </tr>
  <tr></tr> <!-- To disable zebra striping-->
  <tr>
    <td><a href="https://aihub.cloud.google.com/u/0/p/products%2Fffd9bb2e-4917-4c80-acad-67b9427e5fde">Using Tensorflow's Feature Column API for Feature Engineering</a></td>
  </tr>
  <tr></tr> <!-- To disable zebra striping-->

  <!-- To disable zebra striping-->
  <tr></tr>
  <tr></tr>

  <!--Module 3 links-->
  <tr>
    <td rowspan=14 align="center">3</td>
    <td rowspan=14>Design and Build an Input Data Pipeline</td>
    <td><a href="https://medium.com/ml-book/demonstration-of-tensorflow-feature-columns-tf-feature-column-3bfcca4ca5c4">Demonstration of TensorFlow Feature Columns (tf.feature_column)</a></td>
  </tr>
  <tr></tr> <!-- To disable zebra striping-->
  <tr>
    <td><a href="https://aihub.cloud.google.com/u/0/p/products%2Fffd9bb2e-4917-4c80-acad-67b9427e5fde">Using Tensorflow's Feature Column API for Feature Engineering</a></td>
  </tr>
  <tr></tr> <!-- To disable zebra striping-->
  <tr>
    <td><a href="https://www.tensorflow.org/guide/data">tf.data: Build TensorFlow input pipelines</a></td>
  </tr>
  <tr></tr> <!-- To disable zebra striping-->
  <tr>
    <td><a href="https://www.youtube.com/watch?v=kVEOCfBy9uY">Inside TensorFlow: tf.data - TF Input Pipeline</a></td>
  </tr>
  <tr></tr> <!-- To disable zebra striping-->
  <tr>
    <td><a href="https://www.tensorflow.org/datasets/overview">TensorFlow Datasets</a></td>
  </tr>
  <tr></tr> <!-- To disable zebra striping-->
  <tr>
    <td><a href="https://www.youtube.com/watch?v=ZnukSLKEw34">Inside TensorFlow: tf.data + tf.distribute</a></td>
  </tr>
  <tr></tr> <!-- To disable zebra striping-->
  <tr>
    <td><a href="https://www.youtube.com/watch?v=vPrSca-YjFg&list=PU0rqucBdTuFTjJiefW5t-IQ&index=52">Designing a neural network | Text Classification Tutorial Pt. 2 (Coding TensorFlow)</a></td>
  </tr>

  <!-- To disable zebra striping-->
  <tr></tr>

  <!--Module 4 links-->
  <tr>
    <td rowspan=30 align="center">4</td>
    <td rowspan=30>Building Neural Networks with the TensorFlow and Keras API</td>
    <td><a href="https://www.youtube.com/watch?v=VwVg9jCtqaU">Machine Learning - Zero to Hero</a></td>
  </tr>
  <tr></tr> <!-- To disable zebra striping-->
  <tr>
    <td><a href="https://www.youtube.com/watch?v=5ECD8J3dvDQ">Introduction to TensorFlow 2.0: Easier for beginners, and more powerful for experts (TF World '19)</a></td>
  </tr>
  <tr></tr> <!-- To disable zebra striping-->
  <tr>
    <td><a href="https://machinelearningmastery.com/keras-functional-api-deep-learning/">How to Use the Keras Functional API for Deep Learning</a></td>
  </tr>
  <tr></tr> <!-- To disable zebra striping-->
  <tr>
    <td><a href="https://www.pyimagesearch.com/2019/10/28/3-ways-to-create-a-keras-model-with-tensorflow-2-0-sequential-functional-and-model-subclassing/">3 ways to create a Keras model with TensorFlow 2.0 (Sequential, Functional, and Model Subclassing)</a></td>
  </tr>
  <tr></tr> <!-- To disable zebra striping-->
  <tr>
    <td><a href="https://www.youtube.com/watch?v=UYRBHFAvLSs">Tf.keras - part 1</a></td>
  </tr>
  <tr></tr> <!-- To disable zebra striping-->
  <tr>
    <td><a href="https://www.youtube.com/watch?v=uhzGTijaw8A">Tf.keras - part 2</a></td>
  </tr>
  <tr></tr> <!-- To disable zebra striping-->
  <tr>
    <td><a href="https://keras.io/guides/functional_api/">The Keras Functional API</a></td>
  </tr>
  <tr></tr> <!-- To disable zebra striping-->
  <tr>
    <td><a href="https://keras.rstudio.com/articles/functional_api.html">Guide to the Functional API</a></td>
  </tr>
  <tr></tr> <!-- To disable zebra striping-->
  <tr>
    <td><a href="https://medium.com/datadriveninvestor/developing-with-keras-functional-api-6017828408cd">Developing with the Keras Functional API</a></td>
  </tr>
  <tr></tr> <!-- To disable zebra striping-->
  <tr>
    <td><a href="https://developers.google.com/machine-learning/crash-course/regularization-for-simplicity/l2-regularization">Google: Regularization for Simplicity</a></td>
  </tr>
  <tr></tr> <!-- To disable zebra striping-->
  <tr>
    <td><a href="https://developers.google.com/machine-learning/glossary#overfitting">Google Machine Learning Glossary</a></td>
  </tr>
  <tr></tr> <!-- To disable zebra striping-->
  <tr>
    <td><a href="https://www.youtube.com/watch?v=Q81RR3yKn30">Regularization Clearly Explained</a></td>
  </tr>
  <tr></tr> <!-- To disable zebra striping-->
  <tr>
    <td><a href="https://www.youtube.com/watch?v=NGf0voTMlcs">Lasso and Ridge Regression</a></td>
  </tr>
  <tr></tr> <!-- To disable zebra striping-->
  <tr>
    <td><a href="https://www.youtube.com/watch?v=Q81RR3yKn30">Ridge Regression</a></td>
  </tr>
  <tr></tr> <!-- To disable zebra striping-->
  <tr>
    <td><a href="https://machinelearningmastery.com/early-stopping-to-avoid-overtraining-neural-network-models/">A Gentle Introduction to Early Stopping to Avoid Overtraining Neural Networks</a></td>
  </tr>

  <!-- To disable zebra striping-->
  <tr></tr>

  <!--Module 5 links-->
  <tr>
    <td rowspan=8 align="center">5</td>
    <td rowspan=8>Training at Scale with Vertex AI</td>
    <td><a href="https://www.youtube.com/watch?v=v4OZzDlv3aI">Train TensorFlow Models at Scale</a></td>
  </tr>
  <tr></tr> <!-- To disable zebra striping-->
  <tr>
    <td><a href="https://www.youtube.com/watch?v=6ovfZW8pepo">Scaling TensorFlow 2 models to multi-worker GPUs more powerful for experts (TF World '19)</a></td>
  </tr>
  <tr></tr> <!-- To disable zebra striping-->
  <tr>
    <td><a href="https://cloud.google.com/ai-platform/training/docs/training-at-scale">Training at Scale</a></td>
  </tr>
  <tr></tr> <!-- To disable zebra striping-->
  <tr>
    <td><a href="https://colab.sandbox.google.com/github/tensorflow/docs/blob/master/site/en/guide/distributed_training.ipynb">Distributed Training with TensorFlow</a></td>
  </tr>

</table>
