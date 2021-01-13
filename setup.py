#!/usr/bin/env python

from distutils.core import setup

setup(name = 'WormAnalysis',
      version = '1.0',
      description = 'Analysis and plotting functions for \'Decoding locomotion from population neural activity in moving C. elegans\'',
      author = 'Monika Scholz',
      author_email = 'monika.k.scholz@gmail.com',
      packages=['prediction', 'utility'],
      )
