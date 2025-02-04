from __future__ import annotations
from .Image import *
from collections.abc import MutableMapping

class Images(MutableMapping):
  def __init__(self, images, id = 'A0001', seg = None):
    if type(images) not in [list, np.ndarray]: images = [images]
    for i in range(len(images)): 
      if type(images[i]) is str: images[i] = Image(images[i])
      elif type(images[i]) is not Image: raise TypeError('All input images must be of type Image')

    self.images = dict()
    self.id = id
    for image in images: 
      self.images[image.key] = image
      image.id = self.id

    if seg is not None: self.set_seg(seg)
    self.transformed = False

  def set_seg(self, seg):
    if type(seg) is str: seg = Segmentation(seg)
    for image in self.images.values(): 
      image.set_seg(seg)
      if image.mask: image.mask_image()

  def mask_image(self, key = 'scan', new_key = 'brain', normalise = True, centre = None, cmap = 'inferno'):
    self.images[new_key] = Image(new_key, self.images[key].array, self.images[key].seg, self.id, True, normalise = normalise, centre = centre, cmap = cmap)

  def transform(self, transform : str, fix : Union[ants.core.ants_image.ANTsImage, Array3D, Image, Images]):
    if self.transformed: 
      print('Warning: Image has already been transformed')
      return
    if type(fix) == Images: fix = fix.images[list(fix.images.keys)[0]]
    if type(fix) != ants.core.ants_image.ANTsImage: fix = fix.ants
    for im in self.images.values(): im.transform(transform, fix)
    list(self.images.values())[0].seg.transform(transform, fix)


  def register(self, fix : Union[ants.core.ants_image.ANTsImage, Array3D, Image, Images], key = 'brain', return_result = False):
    if type(fix) not in [ants.core.ants_image.ANTsImage, Array3D, Image]: fix = fix.images[key]
    if type(fix) != ants.core.ants_image.ANTsImage: fix = fix.ants
    result = self.images[key].register(fix, return_result = True)
    if return_result: return result
    self.transform(result['fwdtransforms'], fix)

  def __getitem__(self, key) -> Image:
    return self.images[key]
  
  def __setitem__(self, key, image : Image):
    self.images[key] = image
    
  def __iter__(self):
    return iter(self.images)

  def keys(self):
    return self.images.keys()

  def values(self):
    return self.images.values()
  
  def items(self):
    return self.images.items()

  def __delitem__(self, key):
    del self.images[key]
  
  def __len__(self):
    return len(self.images)
  
  def __dir__(self):
    # Include dictionary keys in autocompletion
    return super().__dir__() + list(self.images.keys())
  
  def __getattr__(self, name) -> Image:
    if name in self.images:
        return self.images[name]
    raise AttributeError(f"'Images' object has no attribute '{name}'")