from .Image import *

class Jacobian(Image):
  def __init__(self, data):
    if type(data) == dict:
      data['fwdtransforms']
      


abcd = '/Users/work/Desktop/MPhys/abcd/'

abcd = Database(kf = {
  'scan' : '/Users/work/Desktop/MPhys/abcd/scans',
  'seg' : '/Users/work/Desktop/MPhys/abcd/segmentations',
})

a = abcd.A0009_1
b = abcd.A0009_2

res = a.brain.register(b.brain, True)

print(res['fwdtransforms'])

import ants

arr = ants.create_jacobian_determinant_image(b.brain.ants, res['fwdtransforms'][0])

j = Image('jac', arr.numpy(), seg = b.seg.array, mask = True, centre = 1, normalise=True, cmap = ['white', 'blue', 'darkblue', 'black', 'darkred', 'red', 'white'])