import SimpleITK as sitk
import os
import numpy as np

class Subject:
  def __init__(self, seg_file, age = 0, gender = 'Unknown'):
    self.seg_file = seg_file
    seg = sitk.ReadImage(os.path.join(self.seg_file))
    self.spacing = seg.GetSpacing()
    self.pixel_vol = seg.GetSizeOfPixelComponent()/1000
    seg = sitk.GetArrayFromImage(seg)
    self.total_volume = np.sum(seg.astype(bool).astype(int))*self.pixel_vol
    structures, counts = np.unique(seg, return_counts = True)
    self.vols = dict()
    counts = counts*self.pixel_vol
    for s, c in zip(structures, counts): self.vols[s] = c
    self.id = os.path.split(self.seg_file)[1][:-4]
    self.age, self.gender = age, gender

  def __getitem__(self, key):
    return self.vols[key]
  
  def __str__(self):
    return f'{self.id}'
  
  def print(self, verbose = False):
    print(f'{self.id[:5]} | {"Girl" if self.gender == "F" else " Boy"} | {' ' if self.age < 120 else ''}{self.age // 12} years {' ' if self.age % 12 < 10 else ''}{self.age % 12} months | {' ' if self.total_volume < 1000 else ''}{self.total_volume:.1f}cm\u00b3 | {self.seg_file}')
