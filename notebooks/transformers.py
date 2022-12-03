import numpy as np
class PyTorchImageToNPArray(object):
  """Convert Tensor image which is Color x Height x Width to numpy array of Height x Width x Color"""
  def __call__(self, tensor):
    # tensor, label, filename = sample
    image = np.array(tensor)
    if image.shape[0] == 3:
      # Convert Channel, H, W to H, W, Channel.
      image = image.transpose((1, 2, 0))
    elif image.shape[0] == 1:
      image = image.squeeze(0)
    elif image.shape[2] == 1:
      image = image.squeeze(2)

    return image