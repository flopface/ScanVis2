from .Images import *
from .Scan import *
from .Brain import *
from .Segmentation import *
import numpy as np
from os import listdir
from os.path import join, split, isdir, isfile
from collections.abc import MutableMapping

class Database(MutableMapping):
  def __init__(self, keys = None, folders = None, kf : dict = None):
    if (keys is None or folders is None) and (kf is None): raise TypeError('Either kf, or keys AND folders, must be specified')
    if kf is None:
      if type(keys) in [list, np.ndarray]: keys = [keys]
      if type(folders) in [list, np.ndarray]: folders = [folders]
      if len(keys) != len(folders): print('Warning : Distinct number of keys and folders')
      kf = {}
      for key, folder in zip(keys, folders): kf[key] = folder

    self.kf = kf
    self.validate_folders()

  def validate_folders(self):
    failed = False
    for folder in self.kf.values():
      if not isdir(folder): 
        failed = True
        print(f'{folder} does not exist')
    if failed: raise Exception('All folders must exist')
    self.trim_excess()

  def trim_excess(self):
    self.ids = set([file[:file.find('.')] for file in listdir(list(self.kf.values())[0]) if file[0] != '.'])
    for folder in list(self.kf.values())[1:]:
      ids = set([file[:file.find('.')] for file in listdir(folder) if file[0] != '.'])
      self.ids = self.ids.intersection(ids)
    self.ids = sorted(list(self.ids))
    self.dictionary = dict(zip(ids, [None]*len(ids)))

  def find_file(self, folder, id):
    for file in listdir(folder): 
      if id in file: return join(folder, file)

  def __call__(self, id):
    if ('scan' in self.kf) and ('seg' in self.kf or 'segmentation' in self.kf):
      return self.scan(id)
    return self.image(id)

  def find_id(self, id) -> str:
    if type(id) is not str: id = str(id)
    options = [proper_id for proper_id in self.ids if id in proper_id]
    if len(options) == 0: raise Exception(f'No data with that ID')
    elif len(options) == 1: return options[0]
    else: return user_decision(options)

  def brain(self, id, key = 'scan', seg_key = 'seg') -> Brain:
    if (key in self.kf):
      if seg_key in self.kf:
        proper_id = self.find_id(id)
        return Brain(self.find_file(self.kf[key], proper_id), Segmentation(self.find_file(self.kf[seg_key], proper_id)), id, mask = True)
      else: raise Exception('No segmentations labelled in Database')
    else: raise Exception('No scans labelled in database')

  def array3d(self, proper_id, key) -> Array3D:
    return Array3D(self.find_file(self.kf[key], proper_id))

  def images(self, id) -> Images:
    proper_id = self.find_id(id)
    seg = None
    images = []
    for key, folder in self.kf.items():
      if key in ['seg', 'segmentation']: seg = Segmentation(self.find_file(folder, file))
      else: images.append(Image(key, self.find_file(folder, proper_id)))
    return Images(images, id, seg)
  
  def scan(self, id) -> Scan:
    if ('scan' in self.kf):
      if 'seg' in self.kf or 'segmentation' in self.kf:
        proper_id = self.find_id(id)
        return Scan(self.find_file(self.kf['scan'], proper_id), self.find_file(self.kf['seg' if 'seg' in self.kf else 'segmentation'], proper_id), id)
      else: raise Exception('No segmentations labelled in Database')
    else: raise Exception('No scans labelled in database')

  def standardise_space(self, id : str, save_folder : str):
    fix = self.brain(id)
    for proper_id in self.ids:
      self.brain(proper_id).register
      return

  def timepoint_warp(self, fix_key : str, mov_key : str, supress_print = False, save_folder = None, seg_folder = None, jacobian_folder = None, transform_folder = None):
    if fix_key not in self.kf.keys(): raise FileNotFoundError('Fixed folder not found in database')
    if mov_key not in self.kf.keys(): raise FileNotFoundError('Moving folder not found in database')
    return_result = transform_folder is None

    if jacobian_folder is None and transform_folder is None and save_folder is None: raise Exception('No output folder given, nothing will happen dummy')

    for i, proper_id in enumerate(self.ids):
      if not supress_print: progress_word(i, len(self.ids), 'YOU ARE REALLY HANDSOME BRO')
      fix = self.array3d(proper_id, fix_key)
      mov = self.array3d(proper_id, mov_key)
      result = mov.register(fix, return_result)
      if save_folder is not None: 
        mov.transform(result['fwdtransforms'], fix)
        mov.save(os.path.join(save_folder, proper_id))
      if jacobian_folder is not None:
        mov.array = ants.create_jacobian_determinant_image(fix.ants, result['fwdtransforms'][0]).numpy()
        mov.save(os.path.join(jacobian_folder, proper_id))
      if transform_folder is not None:
        os.makedirs(os.path.join(transform_folder, proper_id))
        os.path.rename(result['fwdtransforms'][0], os.path.join(transform_folder, proper_id, 'warp' if 'warp' in result['fwdtransforms'][0].lower() else 'affine'))
        os.path.rename(result['fwdtransforms'][1], os.path.join(transform_folder, proper_id, 'warp' if 'warp' in result['fwdtransforms'][1].lower() else 'affine'))
    if not supress_print: progress_word(i, len(self.ids), 'YOU ARE REALLY HANDSOME BRO')
        
  def __getitem__(self, name) -> Image:
    if name in self.dictionary:
      if self.dictionary[name] is None: self.dictionary[name] = self.__call__(name)
      return self.dictionary[name]
    raise AttributeError(f"Database object has no attribute '{name}'")
  
  def __setitem__(self, key, image : Image):
    self.dictionary[key] = image
    
  def __iter__(self):
    return iter(self.dictionary)

  def keys(self):
    return self.dictionary.keys()

  def values(self):
    return self.dictionary.values()
  
  def items(self):
    return self.dictionary.items()

  def __delitem__(self, key):
    del self.dictionary[key]
  
  def __len__(self):
    return len(self.dictionary)
  
  def __dir__(self):
    # Include dictionary keys in autocompletion
    return super().__dir__() + list(self.dictionary.keys())
  
  def __getattr__(self, name) -> Image:
    if name in self.dictionary:
      if self.dictionary[name] is None: self.dictionary[name] = self.__call__(name)
      return self.dictionary[name]
    raise AttributeError(f"Database object has no attribute '{name}'")