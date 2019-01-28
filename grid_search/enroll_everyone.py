#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Thu 28 May 2015 10:06:13 CEST

"""This script is called to enroll all people in the database.

First, we create the background model using the training set. Then, we create
individual models for all people in both the development and test sets.
"""

__epilog__ = """example usage:

  To execute the example project:

    $ %(prog)s

  To execute your own project, that lives in module "myproject.py":

    $ %(prog)s myproject

  To change the output directory for your models and execute your own project:

    $ %(prog)s --work-directory=mymodels myproject

"""

import os
import sys
import argparse

def main():

  parser = argparse.ArgumentParser(description=__doc__, epilog=__epilog__,
      formatter_class=argparse.RawDescriptionHelpFormatter)

  parser.add_argument('-w', '--work-directory', metavar='DIR', default='tmp',
      help='Path to the working directory to use for storing files (defaults to `%(default)s\')')

  parser.add_argument('module', default='project', nargs='?', metavar='MODULE',
      help='Name of the module containing the code to execute')

  args = parser.parse_args()

  if not os.path.exists(args.work_directory):
    print("Creating directory `%s'..." % args.work_directory)
    os.makedirs(args.work_directory)
  else:
    print("Using existing directory `%s'..." % args.work_directory)

  print("Loading your project from `%s'..." % args.module)
  exec 'from %s import create_background_model, enroll' % args.module

  from data import get_training_data, get_data

  background_filename = os.path.join(args.work_directory, 'background.model')
  print("Creating background model -> `%s'..." % background_filename)
  if os.path.exists(background_filename): os.unlink(background_filename)
  create_background_model(get_training_data(), background_filename)

  print("Training models...")

  for group in ('devel', 'test'):
    print("... for `%s' group ..." % group)
    data = get_data(group, 'enroll')
    for identity, images in data.iteritems():
      filename = os.path.join(args.work_directory, 'client-%d.model' % identity)
      if os.path.exists(filename): os.unlink(filename)
      print("Enrolling client %d -> `%s'..." % (identity, filename))
      enroll(images, background_filename, filename)

  print("All done. Models saved at directory `%s'." % args.work_directory)
  print("You can proceed with the evaluation using `compute_performance.py'.")

if __name__ == '__main__':
  main()
