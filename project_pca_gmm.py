#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Tue 14 May 2013 09:14:35 CEST

"""This file should contain your project. If this file depends on other files
of your own, you should make sure we also receive a copy of those.

Before starting to edit this file, make sure you read our README.rst file on
the same directory.

In this file we include an example face recognition system you can use as a
baseline for your own project. It implements the Eigen-Faces technique. Here is
the workflow:

  1. We first create a background model using the available data. The
  background model for eigen-faces consists of the first 5 PCA components
  extracted from the flattened images (2D converted to 1D by scanning every row
  of the image in sequence).

  2. To create the Machine for each person, we take the background model and
  project each of the pictures for that given person over the background model
  and average the results. The machine is now trained for the client.

  3. To probe a new picture, we project such a picture over the background
  model again and calculate the L2-norm (a.k.a. Euclidean distance) between the
  projected value and the model. The L2-norm defines the closeness between the
  user model and the picture to be evaluated. The closer they are, the smaller
  the score. To make sure that matching model and picture identities give
  higher scores than in unmatching conditions, we multiply every score by -1.0.

"""

import numpy
import bob.learn.linear
import bob.learn.em
import bob.io.base
import bob.math

def create_background_model(images, filename):
  """Creates the background model

  This function saves the PCA background model to the filename given, using
  HDF5, like we saw in class.

  Keyword arguments:

  images [IN]
    A python iterable (e.g. a list or tuple) that contains all images that are
    available for training the background model. Each image is represented by a
    2D numpy.ndarray composed of 64-bit floats. Each value represents the pixel
    gray level on the face image, with a range from [0.0, 1.0]. The image is
    organized in C-style (or row-wise). Each row in the array represents a row
    of pixels in the original image.

  filename [OUT]
    This is the filename that you MUST use to save your background model to. If
    you modify this variable before saving the file containing the model, it
    will not be possible to retrieve it later on.
  """
  global feature_count	 # n_pc number of PCA components retained
  feature_count = 100

  # perform training using a SVD PCA trainer
  pca_trainer = bob.learn.linear.PCATrainer(use_svd=True)

  # iterates through the training examples and linearize the images
  data = numpy.vstack([image.flatten() for image in images])

  # training the SVD PCA returns a machine that can be used for projection
  pca_machine, eigen_values = pca_trainer.train(data)

  # limits the number of kept eigenfaces (we only use 5 in this example)
  pca_machine.resize(pca_machine.shape[0], 5)

   # saves the transformation matrix into the given file path
  pca_machine.save(bob.io.base.HDF5File(filename, 'w'))

def enroll(images, background_filename, filename):
  """Creates a model for a given person, using available pictures.

  This function takes as input the images for a given person and the
  background model (inside the file ``background_filename``) and registers a
  specific model for the person on the file named ``filename``.

  For this particular example using Eigen Faces, we save as model, the
  average projection of all images against the background model into the given
  file name.

  Keyword arguments:

  images [IN]
    A python iterable (e.g. a list or tuple) that contains all images that are
    available for training a model for a specific person. Each image is
    represented by a 2D numpy.ndarray composed of 64-bit floats. Each value
    represents the pixel gray level on the face image, with a range from [0.0,
    1.0]. The image is organized in C-style (or row-wise). Each row in the
    array represents a row of pixels in the original image.

  background_filename [IN]
    This is the name of the background file you (may have) used to store the
    background model for this project. If you have not saved any background
    model at all - i.e. if your ``create_background_model()`` function is
    empty, then this file may not exist or be empty. Don't use it in this case.

  filename [OUT]
    This is the filename that you MUST use to save the person's model to. If
    you modify this variable before saving the file containing the model, it
    will not be possible to retrieve it later on.
  """

  # loads our "background" model
  pca_machine = bob.learn.linear.Machine(bob.io.base.HDF5File(background_filename, 'r'))

  # projects the data after linearizing them
  projections = [pca_machine(image.flatten()) for image in images]

  # averages the projections - the result is a numpy.ndarray
  average_projection = numpy.mean(projections, axis=0)

  # saves the average projection for that person into a file
  f = bob.io.base.HDF5File(filename, 'w')
  f.set('model', average_projection)

  # ======================================
  # ===== GMM ============================
  # ======================================
  # Train a GMM model on the obtained projections using EM
  # GMM has one component and is trained over features_count- the retained number of principal components
  gmm = bob.learn.em.GMMMachine(10,5)
  trainer = bob.learn.em.ML_GMMTrainer(True, True, True) # update means/variances/weights at each iteration
  bob.learn.em.train(trainer, gmm, projections, max_iterations = 200, convergence_threshold = 1e-5)
  #print(gmm) 

  # Store the GMM model for each enrolled user
  filename = filename + "GMM"
  gmm.save(bob.io.base.HDF5File(filename, "w"))

def probe(images, background_filename, model_filename):
  """Defines the degree of similarity between model and samples

  This function takes as input a set of pre-loaded images, the
  background model (inside the file ``background_filename``) and a particular
  person model (that you created with ``enroll()``) and *returns* a list
  of floating-point values (a.k.a. scores) that represent the degree of
  similarity between that respective image and the client model. For this
  project, we will define similarity like this: high values indicate that the
  picture we are probing and the model **likely** match (i.e., the picture
  belongs to that client).  Lower values indicate that model and probe
  **unlikely** match.

  For this particular example using Eigen Faces, we will use a simple Euclidean
  distance (L2-norm) to compare the projected (new) samples to the loaded model
  for a given user. Because the distance represents how far two vectors are
  from each other, we multiply the result of the distance calculation by -1.0
  so that we conform to the project API.

  Keyword arguments:

  images [IN]
    A python iterable (e.g. a list or tuple) that contains all images that we
    will be test for. Each image is represented by a 2D numpy.ndarray composed
    of 64-bit floats. Each value represents the pixel gray level on the face
    image, with a range from [0.0, 1.0]. The image is organized in C-style (or
    row-wise). Each row in the array represents a row of pixels in the original
    image.

  background_filename [IN]
    This is the name of the background file you (may have) used to store the
    background model for this project. If you have not saved any background
    model at all - i.e. if your ``create_background_model()`` function is
    empty, then this file may not exist or be empty. Don't use it in this case.

  model_filename [IN]
    This is the filename where you saved the model for a particular person. We
    shall use this model to compare the new samples to, using some similarity
    measurement.  """

  # loads our "background" model
  pca_machine = bob.learn.linear.Machine(bob.io.base.HDF5File(background_filename, 'r'))

  # loads the person's specific model
  f = bob.io.base.HDF5File(model_filename, 'r')
  model = f.read('model')

  # read also the GMM file for that person we are comparing against
  model_filename = model_filename + "GMM"
  f = bob.io.base.HDF5File(model_filename, 'r')
  GMMmodel = bob.learn.em.GMMMachine(f) 
  #print(GMMmodel)

  # probe using the log-likelihood
  probes = [pca_machine(image.flatten()) for image in images]
  log_likelihoods = [GMMmodel(probe) for probe in probes]

  # now change the return function
  return  [-1*numpy.linalg.norm(ll) for ll in log_likelihoods]
