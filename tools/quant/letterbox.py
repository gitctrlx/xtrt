from torchvision.transforms import functional as F
from PIL import Image
import torch

class Letterbox:
    def __init__(self, size, fill=(128, 128, 128), mode='bilinear'):
        """
        Letterbox transform for image tensors.
        
        Parameters:
            size (tuple): The target size as (width, height)
            fill (tuple): The color fill for area outside the resized image, default is gray (128, 128, 128).
            mode (str): Interpolation mode to resize the image. Default is 'bilinear'.
                        Options are ['nearest', 'bilinear', 'bicubic'] and possibly others depending on the version of torchvision.
        """
        self.size = size
        self.fill = fill
        self.mode = mode

    def __call__(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be letterboxed.

        Returns:
            PIL Image or Tensor: Letterboxed image.
        """
        # Convert to PIL Image if it's a tensor
        if isinstance(img, torch.Tensor):
            img = F.to_pil_image(img)

        # Original dimensions
        orig_width, orig_height = img.size
        target_width, target_height = self.size

        # Compute scaling factor and output size
        scale = min(target_height / orig_height, target_width / orig_width)
        new_width = int(orig_width * scale)
        new_height = int(orig_height * scale)

        # Resize the image with the specified mode
        img_resized = img.resize((new_width, new_height), Image.Resampling.BILINEAR if self.mode == 'bilinear' else Image.Resampling.NEAREST)

        # Create a new image with the specified background color and size
        img_letterbox = Image.new("RGB", (target_width, target_height), self.fill)
        # Place the resized image on top of the background
        img_letterbox.paste(img_resized, ((target_width - new_width) // 2, (target_height - new_height) // 2))

        return img_letterbox
