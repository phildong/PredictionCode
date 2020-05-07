#!/usr/bin/env python

"""detect which user is using the machine and change the path to the data accordingly."""
import getpass

def dataPath():
	"""hardcoded or automatic detection of the users' data path."""
	host = socket.gethostname()
	if host== 'phy-leiferfcp':
		path = '/Users/leifer/workspace/PredictionCode/'
	elif host=='nif3004':
		path = '/media/scholz_la/hd2/Data/Data'
	elif host=='Ross-PC':
		path = '/mnt/d/worm_data'
	elif host=='tigressdata2.princeton.edu':
		path = '/projects/LEIFER/PanNeuronal/decoding_analysis/worm_data'
	else:
		raise Exception('This is not a known host. Edit prediction/userTracker.py to add your hostname and data paths.')
	return path

def codePath():
	"""hardcoded or automatic detection of the users' data path."""
	host = socket.gethostname()
	if host== 'phy-leiferfcp':
		path = '/Users/leifer/workspace/PredictionCode/'
	elif host=='Ross-PC':
		path = '/home/srossd/PredictionCode/'
	elif host=='tigressdata2.princeton.edu':
		path = '~/PredictionCode'
	else:
		raise Exception('This is not a known host. Edit prediction/userTracker.py to add your hostname and code paths.')
	return path
