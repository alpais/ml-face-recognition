#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Mon 13 May 2013 13:36:47 CEST

"""A library to load and query the AT&T dataset

You should use this library during your exercise to make sure you don't get the
training, development and testing protocols wrong.
"""

# Defines the training, development and testing protocols
ALL_CLIENTS = set(range(1,41)) #1,2,3...,40
TRAINING_CLIENTS = set([1,5,6,9,11,13,14,17,20,21,24,26,27,30,33,34,36,37,38,39])
DEVEL_CLIENTS = set([2,4,8,10,15,19,23,28,29,40])
TEST_CLIENTS = ALL_CLIENTS - TRAINING_CLIENTS - DEVEL_CLIENTS
ALL_FILES = set(range(1,11)) #1,2,3...,10
ENROLL_FILES = set([2,4,5,7,9])
PROBE_FILES = ALL_FILES - ENROLL_FILES

import os
import bob.io.base
import bob.io.image

# The location of the database
DATABASE = os.path.join(os.path.dirname(__file__), 'data')

def get_images_for_client(client, images=[]):
  """Returns a list with all images for the given client pre-loaded

  This method will read the datafiles contained in ``data`` and will return an
  iterable list of 2D numpy.ndarrays. Each 2D numpy.ndarray corresponds to one
  gray-scaled image loaded from the database, where the pixels are organized in
  C scan order (row-by-row). The pixel values are normalized between 0. and 1.,
  with 1. meaning pure white and 0., pure black.

  Keyword parameters:

  client (integer)
    The client identifier, a number between 1 and 40 (inclusive).

  images (integer, optional)
    The pictures of a particular client to retrieve. This must be a python
    iterable with up to 10 integers (1 to 10 (inclusive)), that defines which
    pictures to retrieve for the client. If this list is empty, then returns all
    pictures for that particular client.

  Returns a list with 2D numpy.ndarrays with ``dtype == float`` (i.e., 64-bit
  floats).
  """

  retrieve = images if images else range(1,11) #1,2,3...10
  retval = [bob.io.base.load(os.path.join(DATABASE, 's%d' % client, '%d.pgm' % k)) for k in retrieve]
  return [k.astype('float64')/255. for k in retval]

def get_training_data():
  """Returns the training data as a list of 2D numpy.ndarrays

  This method returns a list like ``get_images_for_client()`` does.
  """

  retval = []
  for client in TRAINING_CLIENTS: retval += get_images_for_client(client)
  return retval

def get_data(group, purpose):
  """Returns the data for client enrollment or probing as a dictionary of 2D
  numpy.ndarrays.

  This method will read the datafiles contained in ``data`` and will return an
  dictionary of 2D numpy.ndarrays. Each 2D numpy.ndarray corresponds to one
  gray-scaled image loaded from the database, where the pixels are organized in
  the same way as for get_images_for_client().

  The dictionary is organized as follows: the keys are the identities of every
  subject (an integer), the values on dictionary correspond to a list of 2D
  numpy.ndarrays containing all images that you can use to either enroll or
  probe each of the clients.

  Keyword arguments:

  group (string)
    The name of the group for which to retrieve the pictures. This parameter
    can be either 'devel' or 'test'.

  purpose
    The purpose for the group of data you want to retrieve. This parameter can
    be either 'enroll', for retrieving the pictures for enrolling client or
    'probe', for retrieving pictures of a given client that are good for
    probing.
  """

  if group not in ('devel', 'test'):
    raise RuntimeError, 'group argument must be either "devel" or "test"'

  if purpose not in ('enroll', 'probe'):
    raise RuntimeError, 'purpose argument must be either "enroll" or "probe"'

  clients = DEVEL_CLIENTS if group == 'devel' else TEST_CLIENTS
  pictures = ENROLL_FILES if purpose == 'enroll' else PROBE_FILES

  retval = {}
  for client in clients:
    retval[client] = get_images_for_client(client, pictures)

  return retval
