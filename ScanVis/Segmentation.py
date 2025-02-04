from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import os
from ipywidgets import interact, fixed
from skimage.transform import rotate
from typing import Literal
from .useful_stuff import *
from .Array3D import *
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, to_rgb, to_rgba

def find_edges_1D(arr):
  string_arr = ''.join(arr.astype(int).astype(str))
  edge_arr = string_arr.replace('01', '02').replace('10', '20') # Turn any True by a False to a 2
  edge_arr = (np.array(list(edge_arr)).astype(int) - 2).astype(bool)
  return ~edge_arr

def find_edges_2D(arr):
  x_arr = np.array([find_edges_1D(line) for line in arr])
  y_arr = np.swapaxes(np.array([find_edges_1D(line) for line in np.swapaxes(arr, 0, 1)]),0,1)
  edge_arr = x_arr | y_arr
  return edge_arr

def find_structure_edges(image, structure_id):
  image = ~(image - structure_id).astype(bool)
  image = find_edges_2D(image)
  return image

def find_structure_coords(image, structure_id):
  image = find_structure_edges(image, structure_id)
  y, x = np.where(image)
  return x, y

def find_structure_and_outline(image, structure_id):
  image = ~(image - structure_id).astype(bool)
  edges = find_edges_2D(image)
  y, x = np.where(edges)
  return image.astype(int), x, y 

class Segmentation(Array3D):
  def __init__(self, data): super().__init__(data)

  def plot(self, ax, view : Literal['Saggittal', 'Axial', 'Coronal'] = 'Saggittal', slice = 120, structure_id = 0, fontsize = 8, color = 'w', ms = 2, fill_alpha = 0.2, outline_alpha = 1, flipped = False, plot_legend = True, label_start = ''):
    if type(structure_id) is int: structure_id = [structure_id]
    if self.isNone or len(structure_id) == 0: return ax
    picture = self.get_slice(view, slice)
    if type(structure_id) not in [list, np.ndarray]: structure_id = [structure_id]
    if type(color) not in [list, np.ndarray]: color = [color]
    if type(ms) not in [list, np.ndarray]: ms = [ms] * len(structure_id)
    if type(fill_alpha) not in [list, np.ndarray]: fill_alpha = [fill_alpha] * len(structure_id)
    if type(outline_alpha) not in [list, np.ndarray]: outline_alpha = [outline_alpha] * len(structure_id)
    if type(flipped) not in [list, np.ndarray]: flipped = [flipped] * len(structure_id)

    for s, c, m, f, o, fl in zip(structure_id, color, ms, fill_alpha, outline_alpha, flipped):
      c = list(to_rgb(c))
      fill, x, y = find_structure_and_outline(picture, s)
      ax.plot(x, y, 'o', c = c, alpha = o, ms = m)#, label = label_start + lut[s] if s in lut else None)
      ax.plot([], [], 'o', c = c, ms = 5, label = label_start + lut[s] if s in lut else None)
      if f > 0:
        fill_cmap = LinearSegmentedColormap.from_list('my_cmap', [[0,0,0,0], c+[f]], 2)
        if fl: ax.imshow(~fill+2, cmap = fill_cmap, vmin = 0, vmax = 1)
        else: ax.imshow(fill, cmap = fill_cmap, vmin = 0, vmax = 1)

    if plot_legend: ax.legend(labelcolor = 'white', facecolor = 'k', loc = 'upper right', fontsize = fontsize)
    return ax

  def get_mask(self):
    return self.array.astype(bool).astype(int)
  
  def overlay(self, ax, view : Literal['Saggittal', 'Axial', 'Coronal'] = 'Saggittal', slice = 120, structure_id = 0, fontsize = 8, color = 'w', ms = 2, fill_alpha = 0.2, outline_alpha = 1, flipped = False, plot_legend = True):
    if self.isNone: raise Exception('No segmentation supplied')
    picture = self.get_slice(view, slice)

    if type(structure_id) not in [list, np.ndarray]: structure_id = [structure_id]
    if type(color) not in [list, np.ndarray]: color = [color]

    for s, c, m, f, o, fl in zip(structure_id, color, ms, fill_alpha, outline_alpha, flipped):
      c = list(to_rgb(c))
      fill, x, y = find_structure_and_outline(picture, s)
      ax.plot(x, y, 's', c = c, alpha = o, ms = m, label = lut[s] if s in lut else None)
      if f > 0:
        fill_cmap = LinearSegmentedColormap.from_list('my_cmap', [[0,0,0,0], c+[f]], 2)
        if fl: ax[1].imshow(~fill+2, cmap = fill_cmap, vmin = 0, vmax = 1)
        else: ax[1].imshow(fill, cmap = fill_cmap, vmin = 0, vmax = 1)

    if plot_legend and len(structure_id) != 0: ax[1].legend(labelcolor = 'white', facecolor = 'k', markerscale = 2, loc = 'upper right', fontsize = fontsize)
    structures, counts = np.unique(picture, return_counts=True)

    inds = np.argsort(structures)
    ax[0].barh([lut[struct] + ' (' + str(struct) + ')' for struct in structures[inds][1:]], counts[inds][1:], height=0.9, align='center', color='r')
    ax[0].set_title('Relative volumes', c = 'w')
    ax[0].set_facecolor('black')
    ax[0].tick_params(axis='x', colors='white')
    ax[0].tick_params(axis='y', colors='white')
    ax[0].spines['top'].set_color('white')
    ax[0].spines['right'].set_color('white')
    ax[0].spines['bottom'].set_color('white')
    ax[0].spines['left'].set_color('white')
    ax[0].set_xticks([])
    ax[0].yaxis.set_tick_params(labelcolor='white')

    return ax