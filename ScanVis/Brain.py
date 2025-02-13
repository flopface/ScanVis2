from .Image import *

class Brain(Image):
  def __init__(self, data, seg : Segmentation = Segmentation(None), id = None, mask = False, normalise = True, centre = None, cmap = 'inferno'):
    super().__init__('brain', data, seg, id, mask, normalise, centre, cmap)