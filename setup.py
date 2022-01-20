#!/usr/bin/env python

from distutils.core import setup

#INSTALL_REQUIRES = ['xarray >= 0.10.6', ] # to be updated

setup(name='itidenatl',
      description='eNATL60 tidal analysis',
      url='https://github.com/NoeLahaye/ITideNATL',
      packages=['itidenatl', 'itidenatl.tools'],
      )

#      install_requires=INSTALL_REQUIRES,
