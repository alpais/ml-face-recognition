#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Thu 28 May 2015 10:06:19 CEST

"""This script is called to compute the performance of your solution

We collect the scores for all probes respecting the protocol for the database
and then compute the threshold a priori on the development set and apply it to
the test set.
"""

import os
import sys
import argparse
import bob.measure
import numpy
import matplotlib.pyplot as mpl
import settings

__epilog__ = """example usage:

  To compute the performance for the example project:

    $ %(prog)s

  To compute the performance for your own project, that lives in module
  "myproject.py":

    $ %(prog)s myproject

  To compute the performance for your own project, that lives in module
  "myproject.py" and using models store on the directory "mymodels":

    $ %(prog)s --work-directory=mymodels myproject

"""

def calculate_scores(data, background_filename, workdir, probe):
  """Calculates scores given certain data

  Keyword arguments:

  data
    The data, in a dictionary as returned by data.get_data() to be probed.

  background_filename
    The full path to the file containing the background model to use

  workdir
    The work directory, where the user has stored the client models

  probe
    The probing method from the user
  """

  positives = []
  negatives = []

  models = data.keys()
  for model_identity in models:

    model_filename = os.path.join(workdir, 'client-%d.model' % model_identity)
    if not os.path.exists(model_filename):
      raise RuntimeError, "Cannot find model for client %d under `%s'" % \
          (model_identity, model_filename)

    for probe_identity, images in data.iteritems():
      print("Matching model of client %d against images of client %d..." % \
          (model_identity, probe_identity))
      scores = probe(images, background_filename, model_filename)
      if model_identity == probe_identity: positives += scores
      else: negatives += scores

  return negatives, positives

def main():

  parser = argparse.ArgumentParser(description=__doc__, epilog=__epilog__,
      formatter_class=argparse.RawDescriptionHelpFormatter)

  parser.add_argument('-w', '--work-directory', metavar='DIR', default='tmp',
      help='Path to the working directory to use for storing files (defaults to `%(default)s\')')

  parser.add_argument('module', default='project', nargs='?', metavar='MODULE',
      help='Name of the module containing the code to execute')

  args = parser.parse_args()

  print("Loading your project from `%s'..." % args.module)
  exec 'from %s import probe' % args.module

  background_filename = os.path.join(args.work_directory, 'background.model')

  if os.path.exists(background_filename):
    print("Using background model from `%s'..." % background_filename)
  else:
    print("N.B.: Background model not found at `%s'... Proceeding w/o a background model." % background_filename)

  from data import get_data

  print("Calculating probe scores for develpment set...")
  devel_data = get_data('devel', 'probe')
  devel_negatives, devel_positives = calculate_scores(devel_data,
    background_filename, args.work_directory, probe)

  print("Calculating probe scores for test set...")
  test_data = get_data('test', 'probe')
  test_negatives, test_positives = calculate_scores(test_data,
    background_filename, args.work_directory, probe)

  print("Calculating the threshold a priori on the development set...")
  threshold = bob.measure.eer_threshold(devel_negatives, devel_positives)

  print("** Development set analysis **")
  dev_far, dev_frr = bob.measure.farfrr(devel_negatives, devel_positives,
      threshold)

  print(" -> Equal-Error Rate (EER) threshold: %g" % threshold)
  print(" -> At this threshold, the performance of your solution is:")
  print("    * FAR = %g%% (%d falsely-accepted out of %d examples)" % \
      (100.*dev_far, dev_far*len(devel_negatives), len(devel_negatives)))
  print("    * FRR = %g%% (%d falsely-rejected out of %d examples)" % \
      (100.*dev_frr, dev_frr*len(devel_positives), len(devel_positives)))
  eer = 50.*(dev_far + dev_frr)
  print("    * EER = %g%%" % eer)

  print("** Test set analysis **")
  test_far, test_frr = bob.measure.farfrr(test_negatives, test_positives,
      threshold)

  print(" -> At the EER threshold @ development set, the performance of your solution is:")
  print("    * FAR = %g%% (%d falsely-accepted out of %d examples)" % \
      (100.*test_far, test_far*len(test_negatives), len(test_negatives)))
  print("    * FRR = %g%% (%d falsely-rejected out of %d examples)" % \
      (100.*test_frr, test_frr*len(test_positives), len(test_positives)))
  hter = 50.*(test_far + test_frr)
  print("    * Half-total Error Rate (HTER) = %g%%" % hter)

  settings.variables["dev_far"] = dev_far
  settings.variables["dev_frr"] = dev_frr
  settings.variables["eer"] = eer
  settings.variables["test_far"] = test_far
  settings.variables["test_frr"] = test_frr
  settings.variables["hter"] = hter

  print("Computing score distributions...")
  min_dev = min(devel_negatives + devel_positives)
  max_dev = max(devel_negatives + devel_positives)
  bin_width = (max_dev-min_dev)/20
  mpl.hist(devel_negatives, label='negatives', normed=True, alpha=0.5,
      color='b', bins=numpy.arange(min_dev, max_dev+bin_width, bin_width))
  mpl.hist(devel_positives, label='positives', normed=True, alpha=0.5,
      color='r', bins=numpy.arange(min_dev, max_dev+bin_width, bin_width))
  mpl.xlabel('Scores')
  mpl.ylabel('Normalized Count')
  mpl.title('Score distribution - Development set')
  mpl.grid()
  leg = mpl.legend(loc='best', fancybox=True)
  leg.get_frame().set_alpha(0.5)
  devel_score_filename = 'devel_scores.pdf'
  print("Saving development set score distribution to `%s'..." %
      devel_score_filename)
  mpl.savefig(devel_score_filename)

  mpl.clf()
  min_test = min(test_negatives + test_positives)
  max_test = max(test_negatives + test_positives)
  bin_width = (max_test-min_test)/20
  mpl.hist(test_negatives, label='negatives', normed=True, alpha=0.5,
      color='b', bins=numpy.arange(min_test, max_test+bin_width, bin_width))
  mpl.hist(test_positives, label='positives', normed=True, alpha=0.5,
      color='r', bins=numpy.arange(min_test, max_test+bin_width, bin_width))
  mpl.xlabel('Scores')
  mpl.ylabel('Normalized Count')
  mpl.title('Score distribution - Test set')
  mpl.grid()
  leg = mpl.legend(loc='best', fancybox=True)
  leg.get_frame().set_alpha(0.5)
  test_score_filename = 'test_scores.pdf'
  print("Saving test set score distribution to `%s'..." %
      test_score_filename)
  mpl.savefig(test_score_filename)

  mpl.clf()
  print("Computing DET performance curve...")
  bob.measure.plot.det(devel_negatives, devel_positives, npoints=1000,
      label=r'devel (eer=%g$\%%$)' % eer, linestyle='--', color='blue')
  bob.measure.plot.det(test_negatives, test_positives, npoints=1000,
      label=r'test (hter=%g$\%%$)' % hter, color='black')
  bob.measure.plot.det_axis([0.01, 50, 0.01, 50])
  mpl.grid()
  leg = mpl.legend(loc='best', fancybox=True)
  leg.get_frame().set_alpha(0.5)
  mpl.xlabel('FRR ($\%$)')
  mpl.ylabel('FAR ($\%$)')
  mpl.title('AT&T Face Recognition FSPR Project')
  figure_filename = 'det.pdf'
  print("Saving DET curve to `%s'..." % figure_filename)
  mpl.savefig(figure_filename)
  print("All done - bye!")

if __name__ == '__main__':
  main()
