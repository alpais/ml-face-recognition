#!/usr/bin/env python

# http://stackoverflow.com/questions/477486/python-decimal-range-step-value
def drange(start, stop, step):
  r = start
  while r < stop:
    yield r
    r += step

import settings
import enroll_everyone
import compute_performance

def main():
  f1=open('results', 'w')
  settings.init()

  for pca_components in range(5, 100, 5):
    for threshold in drange(2, 10, 0.5):
      for boost in drange(2, 10, 0.5):
        settings.variables["pca_components"] = pca_components
        settings.variables["threshold"] = threshold
        settings.variables["boost"] = boost
        settings.variables["dev_far"] = 0.0
        settings.variables["dev_frr"] = 0.0
        settings.variables["eer"] = 0.0
        settings.variables["test_far"] = 0.0
        settings.variables["test_far"] = 0.0
        settings.variables["hter"] = 0.0
        print pca_components, threshold, boost
        enroll_everyone.main()
        compute_performance.main()
        # http://stackoverflow.com/questions/9316023/python-print-to-file
        print >>f1, pca_components, threshold, boost, \
                    settings.variables["dev_far"], \
                    settings.variables["dev_frr"], \
                    settings.variables["eer"], \
                    settings.variables["test_far"], \
                    settings.variables["test_frr"], \
                    settings.variables["hter"]

if __name__ == '__main__':
  main()
        