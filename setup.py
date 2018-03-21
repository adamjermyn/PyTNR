#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
	name = 'TNR',
	version = '1.1',
	description = 'Tensor Network Renormalization',
	author = 'Adam S. Jermyn',
	author_email = 'adamjermyn@gmail.com',
	url = 'https://github.com/adamjermyn/mixer',
	classifiers = [
	'Programming Language :: Python :: 3',
	'License :: OSI Approved :: GPLv3 License',
	'Operating System :: Microsoft :: Windows',
	'Operating System :: POSIX',
	'Operating System :: Unix',
	'Operating System :: MacOS'
	],
	packages=find_packages(exclude=['tests','TensorNetwork_obsolete']),
	install_requires=['numpy']
)

