"""Downsample frame images into compact JPEG files for VLM requests."""

from pathlib import Path

from PIL import Image


class ImagePreprocessor:
    """Re-encode frame images as resized JPEG files.

    Full-resolution camera frames produce large base64 request payloads that
    API backends reject (and local vLLM models must spend tokens on). This
    class re-encodes each frame as a JPEG, optionally downscaling it so the
    longest side fits ``max_size`` while preserving the aspect ratio. Images
    are never upscaled.

    Args:
        max_size: Longest side in pixels for output images. When ``None``,
            only re-encode to JPEG without resizing.
        quality: JPEG quality from 1 to 95.
    """

    SUFFIX = ".jpg"

    def __init__(self, max_size: int | None = None, quality: int = 90):
        if max_size is not None and max_size <= 0:
            raise ValueError("max_size must be greater than zero")
        if not 1 <= quality <= 95:
            raise ValueError("quality must be between 1 and 95")
        self.max_size = max_size
        self.quality = quality

    @classmethod
    def from_config(cls, config: dict):
        """Build a preprocessor from pipeline config keys, or None.

        Returns ``None`` when neither ``image_max_size`` nor ``image_quality``
        is configured, so frames are copied through untouched.
        """
        max_size = config.get("image_max_size")
        quality = config.get("image_quality")
        if max_size is None and quality is None:
            return None
        return cls(max_size=max_size, quality=quality if quality is not None else 90)

    def process(self, src: Path, dst: Path) -> Path:
        """Re-encode ``src`` as a JPEG at ``dst`` and return the output path.

        RGB conversion handles RGBA, palette, and grayscale sources. The
        source file is fully decoded before the destination is written, so
        the two paths may even point at the same file.
        """
        with Image.open(src) as image:
            image = image.convert("RGB")
            if self.max_size is not None:
                image.thumbnail((self.max_size, self.max_size), Image.Resampling.LANCZOS)
        image.save(dst, format="JPEG", quality=self.quality)
        return dst
