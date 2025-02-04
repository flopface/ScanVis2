from .Images import *
from .Scan import *
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
    for key, folder in self.kf.items():
      if not isdir(folder): 
        failed = True
        print(f'{folder} does not exist')
    if failed: raise Exception('All folders must exist')
    self.trim_excess()

  def trim_excess(self):
    self.files = set([file[:file.find('.')] for file in listdir(list(self.kf.values())[0]) if file[0] != '.'])
    for folder in list(self.kf.values())[1:]:
      files = set([file[:file.find('.')] for file in listdir(folder) if file[0] != '.'])
      self.files = self.files.intersection(files)
    self.files = sorted(list(self.files))
    self.dictionary = dict(zip(files, [None]*len(files)))

  def find_file(self, folder, id):
    for file in listdir(folder): 
      if id in file: return join(folder, file)

  def __call__(self, id):
    if ('scan' in self.kf) and ('seg' in self.kf or 'segmentation' in self.kf):
      return self.scan(id)
    return self.image(id)

  def image(self, id) -> Image:
    options = [file for file in self.files if id in file]
    if len(options) == 0: raise Exception(f'No data with that ID')
    elif len(options) == 1: file = options[0]
    else: file = user_decision(options)
    seg = None
    images = []
    for key, folder in self.kf.items():
      if key in ['seg', 'segmentation']: seg = Segmentation(self.find_file(folder, file))
      else: images.append(Image(key, self.find_file(folder, file)))
    return Images(images, id, seg)
  
  def scan(self, id) -> Scan:
    if ('scan' in self.kf):
      if 'seg' in self.kf or 'segmentation' in self.kf:
        options = [file for file in self.files if id in file]
        if len(options) == 0: raise Exception(f'No data with that ID')
        elif len(options) == 1: file = options[0]
        else: file = user_decision(options)
        return Scan(self.find_file(self.kf['scan'], file), self.find_file(self.kf['seg' if 'seg' in self.kf else 'segmentation'], file), id)
      else: raise Exception('No segmentations labelled in Database')
    else: raise Exception('No scans labelled in database')

  def __getitem__(self, key) -> Image:
    return self.dictionary[key]
  
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