from __future__ import annotations
from .useful_stuff import *
import numpy as np
from typing import Literal, Union
import SimpleITK as sitk
import os
from skimage.transform import rotate
import ants

class Array3D():
  def __init__(self, data):
    self.read_data_source(data)

  def read_data_source(self, data):
    self.isNone = False
    self.spacing = [1,1,1]
    self.direction = [[1,0,0],[0,-0,-1],[0,-1,0]]
    
    if type(data) is type(None): 
      self.array = np.zeros((256, 256, 256)).astype(int)
      self.origin = [-128,128,128]
      self.isNone = True
    elif type(data) is np.ndarray:
      if len(data.shape) == 3: 
        self.array = data
        self.origin = [-int(self.array.shape[0]/2),int(self.array.shape[1]/2),int(self.array.shape[2]/2)]
      else: raise TypeError(f'Input data must be 3D - current shape = {data.shape}')
    elif type(data) is str:
      if not os.path.isfile(data): raise Exception(f'{data} not found')
      if data[-4:] == '.nii': 
        im = sitk.ReadImage(data)
        self.array = sitk.GetArrayFromImage(im)
        self.spacing = im.GetSpacing()
        self.direction = np.array(im.GetDirection()).reshape(3, 3)
        self.origin = im.GetOrigin()
      elif data[-4:] == '.npy': 
        self.array = np.load(data)
        self.origin = [-int(self.array.shape[0]/2),int(self.array.shape[1]/2),int(self.array.shape[2]/2)]
    else: raise TypeError(f'Input data must be path to .nii or .npy, or an array, not {data}')

  def get_slice(self, view : Literal['Saggittal', 'Axial', 'Coronal'], slice):
    if view == 'Saggittal': picture = rotate(self.array[:,:,255-slice], 270, preserve_range=True, order = 0)
    elif view == 'Axial': picture = np.flip(self.array[:,slice,:])
    else: picture = np.flip(self.array[slice,:,:], 1)
    return picture
  
  def get_slices(self, view : Literal['Saggittal', 'Axial', 'Coronal'], slices, buffer):
    i = 0
    while not np.any(self.get_slice(view, i)): i += 1
    start = i
    i = 255
    while not np.any(self.get_slice(view, i)): i -= 1
    slices = np.linspace(start+buffer[0], i-buffer[1], slices).astype(int)
    return slices
  
  @property
  def ants(self):
    img = ants.from_numpy(self.array)
    img.set_spacing(self.spacing)
    img.set_origin(self.origin)
    img.set_direction(self.direction)
    return img

  @property
  def sitk(self):
    img = sitk.GetImageFromArray(self.array)
    img.SetSpacing(self.spacing)
    img.SetOrigin(self.origin)
    img.SetDirection(self.direction.flatten())
    return img
  
  def transform(self, transform : str, fix : Union[ants.core.ants_image.ANTsImage, Array3D]):
    if type(fix) != ants.core.ants_image.ANTsImage: fix = fix.ants
    self.array = ants.apply_transforms(fixed=fix, moving=self.ants, transformlist=transform,  interpolator='nearestNeighbor').numpy()

  def register(self, fix : Union[ants.core.ants_image.ANTsImage, Array3D], return_result = False):
    if type(fix) != ants.core.ants_image.ANTsImage: fix = fix.ants
    result = ants.registration(
      fixed = fix,
      moving = self.ants,
      type_of_transform = 'SyN',
      aff_shrink_factors=(6, 4, 2, 1),
      aff_smoothing_sigmas=(3, 2, 1, 0)
    )
    if return_result: return result
    self.transform(result['fwdtransforms'], fix)

  def save(self, save: str = '.', nii: bool = None):
    if save == '': save = '.'
    if os.path.isdir(save): save = os.path.join(save, self.id)
    if nii is None:
      if 'nii' in save: nii = True
      else: nii = False
    elif '.nii' not in save and '.npy' not in save: save += '.nii' if nii else '.npy'
    folder = os.path.dirname(save)
    if folder and not os.path.exists(folder): raise FileNotFoundError(f'The folder {folder} does not exist.')
    if nii: sitk.WriteImage(self.sitk, save)
    else: np.save(save, self.array)