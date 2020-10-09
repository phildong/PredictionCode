# -*- coding: utf-8 -*-
"""
Module to stamp plots with current git hash and creation date.
This helps keep track of what the state of the code was in when it produced a plot.

19 July 2019
Andrew Leifer
leifer@princeton.edu
"""

import git

def getStampString():
    """

    :return: string with date, time, hash, url and path of current code
    """
    ## Stamp with code version and date info
    import git
    repo = git.Repo(search_parent_directories=True)
    hash = repo.head.object.hexsha
    gitpath = repoPath = repo._working_tree_dir
    giturl = repo.remotes.origin.url
    gitbranch = repo.active_branch
    from datetime import datetime
    timestamp = datetime.today().strftime('%Y-%m-%d %H:%M')

    return str(timestamp + '\n' + str(hash) + '\n' + giturl + '\n' + gitpath + '\nBranch: ' + str(gitbranch))

import matplotlib.pyplot as plt


def stamp(ax=plt.gca(), x =.1, y =.5,  notes=''):
    """
    :param ax: matplotlib axes h
    :param x: fractional location in the plot
    :param y: fractional location in the plot
    :return:
    """

    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    # place a text box in upper left in axes coords
    try:
        ax.text(x, y, getStampString()+'\n'+notes, transform=ax.transAxes, fontsize=8, verticalalignment='top', bbox=props)
    except:
        ax.text2D(x, y, getStampString()+'\n'+notes, transform=ax.transAxes, fontsize=8, verticalalignment='top', bbox=props)

