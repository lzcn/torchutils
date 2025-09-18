from PIL import Image


class ResizeToSquare:
    """Resize an image to square with white background padded.

    This transform resizes the input image such that its longer side matches
    the target size, preserving aspect ratio, then pads the shorter side with white.

    Args:
        size (int): Target size for the longest edge.

    Example::

        from torchvision import transforms
        from torchutils.transforms import ResizeToSquare

        transform = transforms.Compose([
            ResizeToSquare(224),
            transforms.ToTensor()
        ])
    """

    def __init__(self, size: int):
        if not isinstance(size, int):
            raise TypeError("Size should be an int")
        self.size = size
        self.target_size = (size, size)

    def __call__(self, img: Image.Image) -> Image.Image:
        if not isinstance(img, Image.Image):
            raise TypeError(f"Input must be a PIL Image. Got {type(img)}")

        if img.size == self.target_size:
            return img

        w, h = img.size
        ratio = self.size / max(w, h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        img = img.resize((new_w, new_h), Image.BILINEAR)

        # paste on white background
        new_img = Image.new("RGB", self.target_size, (255, 255, 255))
        new_img.paste(img, ((self.size - new_w) // 2, (self.size - new_h) // 2))
        return new_img

    def __repr__(self):
        return f"{self.__class__.__name__}(size={self.size})"
