#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 17:27:06 2018

@author: chemla
"""
import os
import numpy as np
from matplotlib import animation
import matplotlib.pyplot as plt
from .visualize_plotting import get_class_ids

def check_data(dataset, out='', tasks=None):
    task_relative = not tasks is None
    if not issubclass(type(tasks), list):
        tasks = [tasks]
    if not os.path.isdir(out):
        os.makedirs(out)
    for t, task in enumerate(tasks):
        all_ids, classes = get_class_ids(dataset, task)  if task_relative else ([list(range(dataset.data[0]))],None)
        class_names = {v:k for k, v in dataset.classes[task].items()}
        for n, ids in enumerate(all_ids):
            print('%s...'%class_names[n])
            i = 0;
            fig = plt.figure()
            ax = fig.add_subplot(111)
            xVals = np.linspace(0, 1, dataset.data[0].shape[0])
            line, = ax.plot(xVals, dataset.data[ids[0]])
            plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg'
            ttl = ax.text(.4, 0.085, os.path.basename(dataset.files[ids[0]]), va='center')
            plt.ylim((-1, 1))
            def updatefig(*args):
                nonlocal i
                i += 1
                line.set_data(xVals, dataset.data[ids[i]])
                ttl.set_text(os.path.basename(dataset.files[ids[i]]))
                return line,
            # Set up formatting for the movie files
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=5, metadata=dict(artist='acids.ircam.fr'), bitrate=1800)
            ani = animation.FuncAnimation(fig, updatefig, frames=len(ids)-2, interval=50, blit=True)
            title = out+'/datasetCheck_%s_%s.mp4'%(task, class_names[n]) if task_relative else out+'/datasetCheck.mp4'
            ani.save(title, writer=writer)
    plt.close('all')

