from .Images import *
from os.path import split

class Scan(Images):
  def __init__(self, scan, seg, id : str = None):
    self.scan_file = scan
    self.seg_file = seg
    super().__init__([
      Image('scan', scan),
      Image('brain', scan, mask = True),
      Image('seg', seg),
    ], id, seg)
    self.set_id(id)
  
  def set_id(self, id : str = None):
    if id is None:
      id = split(self.scan_file)[1][:5]
      if id != split(self.seg_file)[1][:5]: print('Warning : Scan ID does not match Segmentation ID')
    self.id = id
    for image in self.values(): image.set_id(self.id)