#!/usr/bin/env python

"""detect which user is using the machine and change the path to the data accordingly."""
import socket
import getpass

dataPaths = {
	'phy-leiferfcp':               '/Users/leifer/workspace/PredictionCode/',
	'Ross-PC':                     '/mnt/d/worm_data',
	'tigressdata2.princeton.edu':  '/projects/LEIFER/PanNeuronal/decoding_analysis/worm_data'
}

codePaths = {
	'phy-leiferfcp':               '/Users/leifer/workspace/PredictionCode/',
	'Ross-PC':                     '/home/srossd/PredictionCode/',
	'tigressdata2.princeton.edu':  '/home/'+getpass.getuser()+'/PredictionCode'
}

def getPath(dict):
	host = socket.gethostname()
	if host in dict:
		return dict[host]
	else:
		raise Exception('This is not a known host. Edit utility/userTracker.py to add your hostname.')

def dataPath():
	return getPath(dataPaths)

def codePath():
	return getPath(codePaths)
