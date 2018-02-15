#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 16:01:20 2018

@author: ajaver
"""

import pandas as pd
import tables
import matplotlib.pylab as plt

def circle(ax, x, y, radius=0.15):
    #https://matplotlib.org/devdocs/gallery/showcase/anatomy.html#sphx-glr-gallery-showcase-anatomy-py
    from matplotlib.patches import Circle
    from matplotlib.patheffects import withStroke
    circle = Circle((x, y), radius, clip_on=False, zorder=10, linewidth=1,
                    edgecolor='green', facecolor=(0, 0, 0, .0125),
                    path_effects=[withStroke(linewidth=1, foreground='green')])
    ax.add_artist(circle)

examples_file = '/Volumes/behavgenom_archive$/Avelino/screening/CeNDR/skel_examples_r.hdf5'

with pd.HDFStore(examples_file, 'r') as fid:
    #all the worm coordinates and how the skeletons matrix related with a given frame is here
    trajectories_data = fid['/trajectories_data']

#select data from a given frame
frame_number = 1

#read image. If you want the image without the masked backgroudn use "/full_data"
img_field = '/mask'

traj_g = trajectories_data.groupby('frame_number')
frame_data = traj_g.get_group(frame_number)

#select the rows that have the skeletons in this frame
#worms that where not succesfully skeletonized will have a -1 here
skel_id = frame_data['skeleton_id'].values
skel_id = skel_id[skel_id>=0]

with tables.File(examples_file, 'r') as fid:
    
    
    img = fid.get_node(img_field)[frame_number]
    
    skel = fid.get_node('/skeleton')[skel_id, :, :]
    cnt1 = fid.get_node('/contour_side1')[skel_id, :, :]
    cnt2 = fid.get_node('/contour_side2')[skel_id, :, :]

plt.figure(figsize=(20,20))
plt.imshow(img, interpolation='none', cmap='gray')

#add all the worms identified
for _, row in frame_data.iterrows():
   cc = plt.Circle((row['coord_x'], row['coord_y']), row['roi_size']/2, lw=2, color='g', fill=False)
   plt.gca().add_artist(cc)

#add all the worms skeletonized
for (ss, cc1, cc2) in zip(skel, cnt1, cnt2):
    plt.plot(ss[:, 0], ss[:, 1], 'r')
    plt.plot(cc1[:, 0], cc1[:, 1], 'tomato')
    plt.plot(cc2[:, 0], cc2[:, 1], color='salmon')