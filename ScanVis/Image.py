from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, fixed
from typing import Literal
from .useful_stuff import *
from .Segmentation import *
from .Array3D import *
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, to_rgb, to_rgba, cnames

cnames = list(cnames)

class Image(Array3D):
  def __init__(self, key : str, data, seg : Segmentation = Segmentation(None), id = None, mask = False, normalise = True, centre = None, cmap = 'inferno'):
    super().__init__(data)
    self.id = key if id is None else id
    self.key = key
    self.normalise_color(normalise, centre)
    self.set_cmap(cmap)
    self.seg = seg
    self.centre = centre
    self.mask = mask
    if self.mask and not self.seg.isNone: self.mask_image()

  def normalise_color(self, normalise = True, centre = None):
    if normalise:
      smallest, biggest = np.min(self.array), np.max(self.array)
      if centre != None:
        if biggest > -smallest: smallest = (2*centre)-biggest
        else: biggest = (2*centre)-smallest
      self.smallest, self.biggest = smallest, biggest
    else: self.smallest, self.biggest = None, None

  def set_cmap(self, cmap, n_alpha = 10):
    if type(cmap) is str:
      colors = eval(f'plt.cm.{cmap}(np.linspace(0, 1, 256))')
      colors[0:n_alpha, -1] = np.logspace(np.log10(0.01),np.log10(1),n_alpha)
      cmap = ListedColormap(colors)
    elif type(cmap) is list:
      for i in range(len(cmap)): cmap[i] = to_rgba(cmap[i])
      cmap = LinearSegmentedColormap.from_list('my_cmap', cmap, N=256)
    self.cmap = cmap

  def check_ax(self, ax, nx, ny, figsize, dpi, pad = 0, w_pad = None, h_pad = None):
    if ax is None:
      ax_exists = False
      fig, ax = plt.subplots(ncols = nx, nrows = ny, figsize = [figsize[0]*nx, figsize[1]*ny], dpi = dpi)
      fig.patch.set_facecolor('black')
      fig.tight_layout(pad = pad, w_pad = w_pad, h_pad = h_pad)
    else:
      ax_exists = True
      fig = None
    return ax, ax_exists, fig

  def set_seg(self, seg : Segmentation): self.seg = seg

  def set_id(self, id : str): self.id = id

  def mask_image(self):
    mask = self.seg.get_mask()
    self.array = self.array * mask
    if self.centre is not None: self.array += self.centre * (~mask.astype(bool)).astype(int)

  def plot(self, view : Literal['Saggittal', 'Axial', 'Coronal'] = 'Saggittal', slice = 128, structure_id = [], title = None, fontsize = 8, ms = 2, c = 'w', fill_alpha = 0.2, outline_alpha = 1, flipped = False, ax = None, figsize = (5,  5), dpi = 100, save = None, plot_legend = True):
    ax, ax_exists, _ = self.check_ax(ax, 1, 1, figsize, dpi)
    picture = self.get_slice(view, slice)
    ax.imshow(picture, aspect = 1, cmap = self.cmap, vmin = self.smallest, vmax = self.biggest)
    ax = self.seg.plot(ax, view, slice, structure_id, fontsize, c, ms, fill_alpha, outline_alpha, flipped, plot_legend)
    ax.set(yticks = [], xticks = [], frame_on = False)
    ax.set_xlabel(f'Slice {slice} - {self.id} - {self.key.capitalize()} - {view}' if title is None else title, c = 'w', fontsize = fontsize)
    if ax_exists: return ax
    if save != None: plt.savefig(save, dpi = 600, bbox_inches = 'tight')
    plt.show()

  def interactive(self, ms = 2, c = 'w', fill_alpha = 0.2, outline_alpha = 1, flipped = False, title = None, fontsize = 8, figsize = (5, 5), dpi = 100):
    cn = ''.join([f'\'{col}\',' for col in cnames])[:-1]
    if type(c) not in [list, np.ndarray]: c = [c]
    initial_structures = [2, 41, 7, 46, 16, 3, 42, 8, 47] + [1] * 68

    # Setting up strings containing a function to be called, and an interact calling it, with custom inputs to vary the UI
    func_str = f'def plot_str(view, {"".join([f"structure_{i+1} = {initial_structures[i]}, color_{i+1} = \'{cnames[i]}\', " if col is None else f"structure_{i+1} = {initial_structures[i]}, " for i, col in enumerate(c)])}'
    interact_str = f'interact(plot_str, view = [\'Saggittal\', \'Axial\', \'Coronal\'], {"".join([f"structure_{i+1} = (0,77,1), color_{i+1} = [{cn}], " if col is None else f"structure_{i+1} = (0,77,1), " for i, col in enumerate(c)])}'
    # Adding interactive components for ms and alphas if they are input as None
    for item, name, ops in zip([ms, fill_alpha, outline_alpha, fontsize], ['ms = 2', 'fill_alpha = 0.2', 'outline_alpha = 1', 'fontsize = 8'], ['ms = (0,5,0.1)', 'fill_alpha = (0,1,0.01)', 'outline_alpha = (0,1,0.01)', 'fontsize = (4,32,0.1)']):
      if item is None:
        func_str += f'{name}, '
        interact_str += f'{ops}, '
    
    # Finalising strings and placing slice as the final input for cleaner UI
    func_str += 'slice = 100):\n'\
      f'  plot(view, slice, [{"".join([f"structure_{i+1}, " for i in range(len(c))])[:-2]}], title, fontsize, ms, [{"".join([f"color_{i+1}," if col is None else print_color(col)+"," for i, col in enumerate(c)])[:-1]}], fill_alpha, outline_alpha, flipped, None, figsize, dpi, None, True)\n\n'
    interact_str += 'slice = (0,255,1))'

    # Executing the strings to begin the interact
    exec(func_str+interact_str, {'plot' : self.plot, 'interact' : interact, 'title' : title, 'fontsize' : fontsize, 'ms' : ms, 'c' : c, 'fill_alpha' : fill_alpha, 'outline_alpha' : outline_alpha, 'flipped' : flipped, 'figsize' : figsize, 'dpi' : dpi})

  def overlay(self, view : Literal['Saggittal', 'Axial', 'Coronal'] = 'Saggittal', slice = 128, structure_id = [], title = None, fontsize = 8, ms = 2, c = 'w', fill_alpha = 0.2, outline_alpha = 1, flipped = False, ax = None, figsize = (10,  5), dpi = 100, pad = -2, w_pad = None, h_pad = None, save = None, plot_legend = True):
    ax, ax_exists, _ = self.check_ax(ax, 2, 1, figsize, dpi, pad, w_pad, h_pad)
    picture = self.get_slice(view, slice)
    ax[1].imshow(picture, aspect = 1, cmap = self.cmap, vmin = self.smallest, vmax = self.biggest)
    ax = self.seg.overlay(ax, view, slice, structure_id, fontsize, c, ms, fill_alpha, outline_alpha, flipped, plot_legend)
    ax[1].set(yticks = [], xticks = [], frame_on = False)
    ax[1].set_xlabel(f'Slice {slice} - {self.id} - {self.key.capitalize()} - {view}' if title is None else title, c = 'w', fontsize = fontsize)
    if ax_exists: return ax
    if save != None: plt.savefig(save, dpi = 600, bbox_inches = 'tight')
    plt.show()  

  def interactive_overlay(self, ms = 2, c = 'w', fill_alpha = 0.2, outline_alpha = 1, flipped = False, title = None, fontsize = 8, figsize = (10, 5), dpi = 100, pad = -2, w_pad = None, h_pad = None):
    N = len(c) if type(c) != str else 1
    func_str = \
      f'def overlay_str(view, {"".join([f"structure_{i+1} = {i+2}, " for i in range(N)])}slice = 100):\n'\
      f'  overlay(view, slice, [{"".join([f"structure_{i+1}, " for i in range(N)])[:-2]}], title, fontsize, ms, c, fill_alpha, outline_alpha, flipped, None, figsize, dpi, pad, w_pad, h_pad, None, True)\n\n'\
      f'interact(overlay_str, view = [\'Saggittal\', \'Axial\', \'Coronal\'], {"".join([f"structure_{i+1} = (0,77,1), " for i in range(N)])}slice = (0,255,1))'
    
    exec(func_str, {'overlay' : self.overlay, 'interact' : interact, 'title' : title, 'fontsize' : fontsize, 'ms' : ms, 'c' : c, 'fill_alpha' : fill_alpha, 'outline_alpha' : outline_alpha, 'flipped' : flipped, 'figsize' : figsize, 'dpi' : dpi, 'pad' : pad, 'w_pad' : w_pad, 'h_pad' : h_pad})

  def compare(self, other : Image, view : Literal['Saggittal', 'Axial', 'Coronal'] = 'Saggittal', slice = 128, structure_id = [], title = None, fontsize = 8, ms = 2, c = 'w', fill_alpha = 0.2, outline_alpha = 1, flipped = False, ax = None, figsize = (5,  5), dpi = 100, pad = -2, w_pad = None, h_pad = None, save = None, plot_legend = True):
    ax, ax_exists, fig = self.check_ax(ax, 2, 1, figsize, dpi, pad, w_pad, h_pad)
    ax[0]  = self.plot(view, slice, structure_id, title, fontsize, ms, c, fill_alpha, outline_alpha, flipped, ax[0], plot_legend=plot_legend)
    ax[1] = other.plot(view, slice, structure_id, title, fontsize, ms, c, fill_alpha, outline_alpha, flipped, ax[1], plot_legend=plot_legend)
    if title != None and fig != None: fig.suptitle(title, fontsize = fontsize[1], c = 'w')
    if ax_exists: return ax
    if save != None: plt.savefig(save, dpi = 600, bbox_inches = 'tight')
    plt.show()

  def interactive_compare(self, other : Image, ms = 2, c = 'w', fill_alpha = 0.2, outline_alpha = 1, flipped = False, title = None, fontsize = 8, figsize = (5, 5), dpi = 100, pad = -2, w_pad = None, h_pad = None):
    N = len(c) if type(c) != str else 1
    func_str = \
      f'def compare_str(view, {"".join([f"structure_{i+1} = {i+2}, " for i in range(N)])}slice = 100):\n'\
      f'  compare(other, view, slice, [{"".join([f"structure_{i+1}, " for i in range(N)])[:-2]}], title, fontsize, ms, c, fill_alpha, outline_alpha, flipped, None, figsize, dpi, pad, w_pad, h_pad, None, True)\n\n'\
      f'interact(compare_str, view = [\'Saggittal\', \'Axial\', \'Coronal\'], {"".join([f"structure_{i+1} = (0,77,1), " for i in range(N)])}slice = (0,255,1))'
    
    exec(func_str, {'compare' : self.compare, 'other' : other, 'interact' : interact, 'title' : title, 'fontsize' : fontsize, 'ms' : ms, 'c' : c, 'fill_alpha' : fill_alpha, 'outline_alpha' : outline_alpha, 'flipped' : flipped,'figsize' : figsize, 'dpi' : dpi, 'pad' : pad, 'w_pad' : w_pad, 'h_pad' : h_pad})

  def compare_rgb(self, other : Image, view : Literal['Saggittal', 'Axial', 'Coronal'] = 'Saggittal', slice = 128, structure_id = [], title = None, fontsize = 8, ms = 2, c = ['magenta', 'lime'], fill_alpha = 0, outline_alpha = 1, flipped = False, ax = None, figsize = (5,  5), dpi = 100, save = None, plot_legend = True):
    ax, ax_exists, _ = self.check_ax(ax, 1, 1, figsize, dpi)
    rgb = np.zeros((256, 256, 3))
    picture = self.get_slice(view, slice)
    rgb[:,:,0] = picture
    rgb[:,:,2] = picture
    rgb[:,:,1] = other.get_slice(view, slice)
    for i in [0,1,2]: rgb[:,:,i] /= np.max(rgb[:,:,i])*2
    ax.imshow(rgb)
    ax = self.seg.plot(ax, view, slice, structure_id, fontsize, c[::2], ms, fill_alpha, outline_alpha, flipped, False, self.id+' ')
    ax = other.seg.plot(ax, view, slice, structure_id, fontsize, c[1::2], ms, fill_alpha, outline_alpha, flipped, False, other.id+' ')
    ax.set(yticks = [], xticks = [], frame_on = False)
    ax.set_xlabel(f'Slice {slice} - {self.id} - {other.id} - {view}' if title is None else title, c = 'w', fontsize = fontsize)
    if plot_legend and len(structure_id) > 0: ax.legend(labelcolor = 'white', facecolor = 'k', markerscale = 2, loc = 'upper right', fontsize = fontsize)
    if ax_exists: return ax
    if save != None: plt.savefig(save, dpi = 600, bbox_inches = 'tight')
    plt.show()

  def interactive_compare_rgb(self, other : Image, ms = 2, c = 'w', fill_alpha = 0, outline_alpha = 1, flipped = False, title = None, fontsize = 8, figsize = (10, 5), dpi = 100):
    N = int(len(c)/2) if type(c) != str else 1
    func_str = \
      f'def compare_rgb_str(view, {"".join([f"structure_{i+1} = {i+2}, " for i in range(N)])}slice = 100):\n'\
      f'  compare_rgb(other, view, slice, [{"".join([f"structure_{i+1}, " for i in range(N)])[:-2]}], title, fontsize, ms, c, fill_alpha, outline_alpha, flipped, None, figsize, dpi, None, True)\n\n'\
      f'interact(compare_rgb_str, view = [\'Saggittal\', \'Axial\', \'Coronal\'], {"".join([f"structure_{i+1} = (0,77,1), " for i in range(N)])}slice = (0,255,1))'
    
    exec(func_str, {'compare_rgb' : self.compare_rgb, 'other' : other, 'interact' : interact, 'title' : title, 'fontsize' : fontsize, 'ms' : ms, 'c' : c, 'fill_alpha' : fill_alpha, 'outline_alpha' : outline_alpha, 'flipped' : flipped, 'figsize' : figsize, 'dpi' : dpi})

  def plot_three(self, slices = [128, 128, 128], structure_id = [], title = None, fontsize = 8, ms = 2, c = 'w', fill_alpha = 0.2, outline_alpha = 1, flipped = False, ax = None, figsize = (4,4), dpi = 100, pad = -2, w_pad = None, h_pad = None, save = None, plot_legend = True): 
    ax, ax_exists, fig = self.check_ax(ax, 3, 1, figsize, dpi, pad, w_pad, h_pad)
    for i, view in enumerate(['Saggittal', 'Axial', 'Coronal']): ax[i] = self.plot(view, slices[i], structure_id, None, fontsize, ms, c, fill_alpha, outline_alpha, ax[i], plot_legend = i == 2 and plot_legend)
    if title != None and fig != None: fig.suptitle(title, fontsize = fontsize[1], c = 'w')
    if ax_exists: return ax
    if save != None: plt.savefig(save, dpi = 600, bbox_inches='tight')
    plt.show()

  def compare_three(self, other : Image, slices = [128, 128, 128], structure_id = [], title = None, fontsize = 8, ms = 2, c = ['magenta', 'lime'], fill_alpha = 0.2, outline_alpha = 1, flipped = False, ax = None, figsize = (4,4), dpi = 100, pad = -2, w_pad = None, h_pad = None, save = None, plot_legend = True): 
    ax, ax_exists, fig = self.check_ax(ax, 3, 2, figsize, dpi, pad, w_pad, h_pad)
    for i, view in enumerate(['Saggittal', 'Axial', 'Coronal']): ax[0][i] = self.plot(view, slices[i], structure_id, None, fontsize, ms, c, fill_alpha, outline_alpha, flipped, ax[0][i], plot_legend = i == 2 and plot_legend)
    for i, view in enumerate(['Saggittal', 'Axial', 'Coronal']): ax[1][i] = other.plot(view, slices[i], structure_id, None, fontsize, ms, c, fill_alpha, outline_alpha, flipped, ax[1][i], plot_legend = i == 2 and plot_legend)
    if title != None and fig != None: fig.suptitle(title, fontsize = fontsize[1], c = 'w')
    if ax_exists: return ax
    if save != None: plt.savefig(save, dpi = 600, bbox_inches='tight')
    plt.show()

  def compare_three_rgb(self, other : Image, slices = [128, 128, 128], structure_id = [], title = None, fontsize = 8, ms = 2, c = ['magenta', 'lime'], fill_alpha = 0.2, outline_alpha = 1, flipped = False, ax = None, figsize = (4,4), dpi = 100, pad = -2, w_pad = None, h_pad = None, save = None, plot_legend = True): 
    ax, ax_exists, fig = self.check_ax(ax, 3, 1, figsize, dpi, pad, w_pad, h_pad)
    for i, view in enumerate(['Saggittal', 'Axial', 'Coronal']): ax[i] = self.compare_rgb(other, view, slices[i], structure_id, None, fontsize, ms, c, fill_alpha, outline_alpha, flipped, ax[i], plot_legend = i == 2 and plot_legend)
    if title != None and fig != None: fig.suptitle(title, fontsize = fontsize[1], c = 'w')
    if ax_exists: return ax
    if save != None: plt.savefig(save, dpi = 600, bbox_inches='tight')
    plt.show()

  def plot_hella_slices(self, view : Literal['Saggittal', 'Axial', 'Coronal'] = 'Saggittal', slices = 5, buffer = 10, structure_id = [], title = None, fontsize = 8, ms = 2, c = 'w', fill_alpha = 0.2, outline_alpha = 1, flipped = False, ax = None, figsize = (3,3), dpi = 100, pad = -2, w_pad = None, h_pad = None, save = None, label_slices = True, label_images = False, plot_legend = True): 
    ax, ax_exists, fig = self.check_ax(ax, slices, 1, figsize, dpi, pad, w_pad, h_pad)
    if type(buffer) not in [list, np.ndarray]: buffer = [buffer, buffer]
    if type(fontsize) not in [list, np.ndarray]: fontsize = [fontsize, fontsize]
    if type(slices) == int: slices = self.seg.get_slices(view, slices, buffer)
    for i, slice in enumerate(slices): ax[i] = self.plot(view, slices[i], structure_id, f'Slice {slice}' if label_slices else None if label_images else '', fontsize[0], ms, c, fill_alpha, outline_alpha, flipped, ax[i], plot_legend = (i == len(slices)-1) and plot_legend)
    if title != None and fig != None: fig.suptitle(title, fontsize = fontsize[1], c = 'w')
    if ax_exists: return ax
    if save != None: plt.savefig(save, dpi = 600, bbox_inches='tight')
    plt.show()
  
  def compare_hella_slices(self, other : Image, view : Literal['Saggittal', 'Axial', 'Coronal'] = 'Saggittal', slices = 5, buffer = 10, structure_id = [], title = None, fontsize = 8, ms = 2, c = 'w', fill_alpha = 0.2, outline_alpha = 1, flipped = False, ax = None, figsize = (3,3), dpi = 100, pad = -2, w_pad = None, h_pad = None, save = None, label_slices = True, label_images = False, plot_legend = True): 
    ax, ax_exists, fig = self.check_ax(ax, slices, 2, figsize, dpi, pad, w_pad, h_pad)
    if type(buffer) not in [list, np.ndarray]: buffer = [buffer, buffer]
    if type(fontsize) not in [list, np.ndarray]: fontsize = [fontsize, fontsize]
    if type(label_slices) not in [list, np.ndarray]: label_slices = [label_slices, label_slices]
    if type(slices) == int: slices = self.seg.get_slices(view, slices, buffer)
    for i, slice in enumerate(slices): ax[0][i] = self.plot(view, slices[i], structure_id, f'Slice {slice}' if label_slices[0] else None if label_images else '', fontsize[0], ms, c, fill_alpha, outline_alpha, flipped, ax[0][i], plot_legend = (i == len(slices)-1) and plot_legend)
    for i, slice in enumerate(slices): ax[1][i] = other.plot(view, slices[i], structure_id, f'Slice {slice}' if label_slices[1] else None if label_images else '', fontsize[0], ms, c, fill_alpha, outline_alpha, flipped, ax[1][i])
    ax[0][0].set_ylabel(f'{self.id} {view}', color = 'w', fontsize = fontsize[0])
    ax[1][0].set_ylabel(f'{other.id} {view}', color = 'w', fontsize = fontsize[0])
    if title != None and fig != None: fig.suptitle(title, fontsize = fontsize[1], c = 'w')
    if ax_exists: return ax
    if save != None: plt.savefig(save, dpi = 600, bbox_inches='tight')
    plt.show()

  def compare_hella_slices_rgb(self, other : Image, view : Literal['Saggittal', 'Axial', 'Coronal'] = 'Saggittal', slices = 5, buffer = 10, structure_id = [], title = None, fontsize = 8, ms = 2, c = 'w', fill_alpha = 0.2, outline_alpha = 1, flipped = False, ax = None, figsize = (3,3), dpi = 100, pad = -2, w_pad = None, h_pad = None, save = None, label_slices = True, label_images = False, plot_legend = True): 
    ax, ax_exists, fig = self.check_ax(ax, slices, 1, figsize, dpi, pad, w_pad, h_pad)
    if type(buffer) not in [list, np.ndarray]: buffer = [buffer, buffer]
    if type(fontsize) not in [list, np.ndarray]: fontsize = [fontsize, fontsize]
    if type(slices) == int: slices = self.seg.get_slices(view, slices, buffer)
    for i, slice in enumerate(slices): ax[i] = self.compare_rgb(other, view, slices[i], structure_id, f'Slice {slice}' if label_slices else None if label_images else '', fontsize[0], ms, c, fill_alpha, outline_alpha, flipped, ax[i], plot_legend = (i == len(slices)-1) and plot_legend)
    if title != None and fig != None: fig.suptitle(title, fontsize = fontsize[1], c = 'w')
    if ax_exists: return ax
    if save != None: plt.savefig(save, dpi = 600, bbox_inches='tight')
    plt.show()

  def plot_buttloads_of_slices(self, slices = 5, buffer = 10, structure_id = [], title = None, fontsize = 8, ms = 2, c = 'w', fill_alpha = 0.2, outline_alpha = 1, flipped = False, ax = None, figsize = (3, 3), dpi = 100, pad = -2, w_pad = None, h_pad = None, save = None, label_slices = True, label_images = False, plot_legend = True): 
    ax, ax_exists, fig = self.check_ax(ax, slices, 3, figsize, dpi, pad, w_pad, h_pad)
    
    # Allowed buffer inputs are a, [a, b], or [[a, b], [c, d], [e, f]]
    if type(buffer) not in [list, np.ndarray]: buffer = [[buffer, buffer]]*3
    elif len(np.shape(buffer)) == 1: buffer = [buffer]*3
    if type(fontsize) not in [list, np.ndarray]: fontsize = [fontsize, fontsize]
    if type(label_slices) not in [list, np.ndarray]: label_slices = [label_slices, label_slices]

    for i, view in enumerate(['Saggittal', 'Axial', 'Coronal']): ax[i] = self.plot_hella_slices(view, slices, buffer[i], structure_id, None, fontsize, ms, c, fill_alpha, outline_alpha, flipped, ax[i], label_slices = label_slices, label_images=label_images, plot_legend = (i == 0) and plot_legend)
    ax[0][0].set_ylabel('Saggittal', color = 'w', fontsize = fontsize[0])
    ax[1][0].set_ylabel('Axial', color = 'w', fontsize = fontsize[0])
    ax[2][0].set_ylabel('Coronal', color = 'w', fontsize = fontsize[0])
    if title != None and fig != None: fig.suptitle(title, fontsize = fontsize[1], c = 'w')
    if ax_exists: return ax
    if save != None: plt.savefig(save, dpi = 600, bbox_inches='tight')
    plt.show()

  def compare_buttloads_of_slices(self, other : Image, slices = 5, buffer = 10, structure_id = [], title = None, fontsize = 8, ms = 2, c = 'w', fill_alpha = 0.2, outline_alpha = 1, flipped = False, ax = None, figsize = (3, 3), dpi = 100, pad = -2, w_pad = None, h_pad = None, save = None, label_slices = True, label_images = False, plot_legend = True): 
    ax, ax_exists, fig = self.check_ax(ax, slices, 6, figsize, dpi, pad, w_pad, h_pad)
    
    # Allowed buffer inputs are a, [a, b], or [[a, b], [c, d], [e, f]]
    if type(buffer) not in [list, np.ndarray]: buffer = [[buffer, buffer]]*3
    elif len(np.shape(buffer)) == 1: buffer = [buffer]*3
    if type(fontsize) not in [list, np.ndarray]: fontsize = [fontsize, fontsize]

    for i, view in enumerate(['Saggittal', 'Axial', 'Coronal']): ax[2*i:2*i+2] = self.compare_hella_slices(other, view, slices, buffer[i], structure_id, None, fontsize, ms, c, fill_alpha, outline_alpha, flipped, ax[2*i:2*i+2], label_slices = label_slices, label_images=label_images, plot_legend = (i == 0) and plot_legend)
    if title != None and fig != None: fig.suptitle(title, fontsize = fontsize[1], c = 'w')
    if ax_exists: return ax
    if save != None: plt.savefig(save, dpi = 600, bbox_inches='tight')
    plt.show()

  def compare_buttloads_of_slices_rgb(self, other : Image, slices = 5, buffer = 10, structure_id = [], title = None, fontsize = 8, ms = 2, c = 'w', fill_alpha = 0.2, outline_alpha = 1, flipped = False, ax = None, figsize = (3, 3), dpi = 100, pad = -2, w_pad = None, h_pad = None, save = None, label_slices = True, label_images = False, plot_legend = True): 
    ax, ax_exists, fig = self.check_ax(ax, slices, 3, figsize, dpi, pad, w_pad, h_pad)
    
    # Allowed buffer inputs are a, [a, b], or [[a, b], [c, d], [e, f]]
    if type(buffer) not in [list, np.ndarray]: buffer = [[buffer, buffer]]*3
    elif len(np.shape(buffer)) == 1: buffer = [buffer]*3
    if type(fontsize) not in [list, np.ndarray]: fontsize = [fontsize, fontsize]
    if type(label_slices) not in [list, np.ndarray]: label_slices = [label_slices, label_slices]

    for i, view in enumerate(['Saggittal', 'Axial', 'Coronal']): ax[i] = self.compare_hella_slices_rgb(other, view, slices, buffer[i], structure_id, None, fontsize, ms, c, fill_alpha, outline_alpha, flipped, ax[i], label_slices = label_slices, label_images=label_images, plot_legend = (i == 0) and plot_legend)
    ax[0][0].set_ylabel('Saggittal', color = 'w', fontsize = fontsize[0])
    ax[1][0].set_ylabel('Axial', color = 'w', fontsize = fontsize[0])
    ax[2][0].set_ylabel('Coronal', color = 'w', fontsize = fontsize[0])
    if title != None and fig != None: fig.suptitle(title, fontsize = fontsize[1], c = 'w')
    if ax_exists: return ax
    if save != None: plt.savefig(save, dpi = 600, bbox_inches='tight')
    plt.show()