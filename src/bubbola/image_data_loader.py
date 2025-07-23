import base64
import hashlib
import pickle
import time
from collections.abc import Iterable
from datetime import datetime
from io import BytesIO
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image


class CacheManager:
    def __init__(self, cache_dir: Path | None = None, max_age_days: int = 30):
        if cache_dir is None:
            # Try to get home directory, fallback to current directory if it fails
            try:
                home_dir = Path.home()
                self.cache_dir = home_dir / ".cache" / "ai_bubbles"
            except (RuntimeError, OSError):
                # Fallback for CI environments where home directory cannot be determined
                self.cache_dir = Path.cwd() / ".cache" / "ai_bubbles"
        else:
            self.cache_dir = cache_dir

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = self.cache_dir / "cache.pkl"
        self.max_age_days = max_age_days
        self._cache: dict[str, dict] = self._load()
        self._cleanup_old_entries()

    def _load(self) -> dict[str, dict]:
        """Load the cache from disk."""
        if self.cache_path.exists():
            with open(self.cache_path, "rb") as f:
                return pickle.load(f)

        return {}

    def save(self):
        """Save the cache to disk."""
        temp_path = self.cache_path.with_suffix(".tmp")
        with open(temp_path, "wb") as f:
            pickle.dump(self._cache, f)
        temp_path.replace(self.cache_path)

    def get(self, key: str) -> list[bytes] | None:
        """Get a value from cache."""
        entry = self._cache.get(key)
        if entry is None:
            return None

        # Check if entry is expired
        if self._is_expired(entry):
            del self._cache[key]
            return None

        return entry["data"]

    def set(self, key: str, value: list[bytes]):
        """Set a value in cache."""
        self._cache[key] = {
            "data": value,
            "timestamp": time.time(),
            "created": datetime.now().isoformat(),
        }

    def compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA256 hash of a file for cache key."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def compute_bytes_hash(self, data: bytes) -> str:
        """Compute SHA256 hash of bytes for cache key."""
        return hashlib.sha256(data).hexdigest()

    def _is_expired(self, entry: dict) -> bool:
        """Check if a cache entry is expired."""
        if "timestamp" not in entry:
            return True  # Old format entries are considered expired

        age_seconds = time.time() - entry["timestamp"]
        max_age_seconds = self.max_age_days * 24 * 60 * 60
        return age_seconds > max_age_seconds

    def _cleanup_old_entries(self):
        """Remove expired entries from cache."""
        expired_keys = []
        for key, entry in self._cache.items():
            if self._is_expired(entry):
                expired_keys.append(key)

        for key in expired_keys:
            del self._cache[key]

        if expired_keys:
            print(f"Cleaned up {len(expired_keys)} expired cache entries")


class PDFConverter:
    def __init__(self, cache_manager: CacheManager):
        self.cache_manager = cache_manager

    def convert_from_path(self, path: Path) -> list[Image.Image]:
        cache_key = self.cache_manager.compute_file_hash(path)
        cached = self.cache_manager.get(cache_key)
        if cached:
            print(f"Cache HIT for {path.name}")
            return [_deserialize_image(img_bytes) for img_bytes in cached]

        print(f"Cache MISS for {path.name}")
        images = self._convert_pdf_to_images(str(path))
        self.cache_manager.set(cache_key, [_serialize_image(img) for img in images])
        return images

    def convert_from_bytes(self, data: bytes) -> list[Image.Image]:
        cache_key = self.cache_manager.compute_bytes_hash(data)
        cached = self.cache_manager.get(cache_key)
        if cached:
            print("Cache HIT for bytes data")
            return [_deserialize_image(img_bytes) for img_bytes in cached]

        print("Cache MISS for bytes data")
        images = self._convert_pdf_bytes_to_images(data)
        self.cache_manager.set(cache_key, [_serialize_image(img) for img in images])
        return images

    def _convert_pdf_to_images(self, pdf_path: str) -> list[Image.Image]:
        """Convert PDF file to list of PIL Images using PyMuPDF."""
        doc = fitz.open(pdf_path)
        images = []

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            # Render page to image with 2x zoom for better quality
            mat = fitz.Matrix(2.0, 2.0)
            pix = page.get_pixmap(matrix=mat)

            # Convert to PIL Image
            img_data = pix.tobytes("png")
            img = Image.open(BytesIO(img_data))
            images.append(img)

        doc.close()
        return images

    def _convert_pdf_bytes_to_images(self, pdf_bytes: bytes) -> list[Image.Image]:
        """Convert PDF bytes to list of PIL Images using PyMuPDF."""
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        images = []

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            # Render page to image with 2x zoom for better quality
            mat = fitz.Matrix(2.0, 2.0)
            pix = page.get_pixmap(matrix=mat)

            # Convert to PIL Image
            img_data = pix.tobytes("png")
            img = Image.open(BytesIO(img_data))
            images.append(img)

        doc.close()
        return images


class ImageLoader:
    def __init__(self, pdf_converter: PDFConverter, max_edge_size: int | None = None):
        self.pdf_converter = pdf_converter
        self.max_edge_size = max_edge_size

    def load(self, data: str | Path | bytes | Image.Image) -> list[Image.Image]:
        if isinstance(data, str | Path):
            path = Path(data)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")

            if path.suffix.lower() == ".pdf":
                images = self.pdf_converter.convert_from_path(path)
                return [_resize_image(img, self.max_edge_size) for img in images]
            return [_resize_image(Image.open(path), self.max_edge_size)]

        if isinstance(data, bytes):
            # try:
            images = self.pdf_converter.convert_from_bytes(data)
            return [_resize_image(img, self.max_edge_size) for img in images]
            # except:
            #     return [_resize_image(Image.open(BytesIO(data)), self.max_edge_size)]

        if isinstance(data, Image.Image):
            return [_resize_image(data, self.max_edge_size)]

        raise ValueError(f"Unsupported input type: {type(data)}")


class Base64ImageLoader:
    def __init__(self, image_loader: ImageLoader):
        self.image_loader = image_loader

    def load(self, data: str | Path | bytes | Image.Image) -> list[str]:
        images = self.image_loader.load(data)
        return [_image_to_base64(img) for img in images]


def _serialize_image(img: Image.Image) -> bytes:
    """Convert PIL Image to bytes for caching."""
    img_bytes = BytesIO()
    img.save(img_bytes, format="PNG")
    return img_bytes.getvalue()


def _deserialize_image(img_bytes: bytes) -> Image.Image:
    """Convert bytes back to PIL Image."""
    return Image.open(BytesIO(img_bytes))


def _image_to_base64(img: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    img_bytes = BytesIO()
    img.save(img_bytes, format="PNG")
    return base64.b64encode(img_bytes.getvalue()).decode("utf-8")


def _resize_image(img: Image.Image, max_edge_size: int | None = None) -> Image.Image:
    """Resize image maintaining aspect ratio if max_edge_size is specified."""
    if max_edge_size is None:
        return img

    width, height = img.size
    if width <= max_edge_size and height <= max_edge_size:
        return img

    if width > height:
        new_width = max_edge_size
        new_height = int(height * (max_edge_size / width))
    else:
        new_height = max_edge_size
        new_width = int(width * (max_edge_size / height))

    return img.resize((new_width, new_height), Image.Resampling.LANCZOS)


def sanitize_to_images(
    input_data: str | Path | bytes | Image.Image | Iterable | None,
    return_as_base64: bool = False,
    max_edge_size: int | None = None,
) -> dict[str, Image.Image | str]:
    """Convert various input types into a dictionary of PIL Images or base64 strings.

    Handles:
    - Path/string to PDF, image file
    - Path/string to folder containing PDFs and images
    - Raw bytes of PDF or image
    - PIL Image objects
    - Iterables containing any of the above

    Uses persistent caching for all processing steps (including resizing and base64) to improve performance on subsequent runs.

    Args:
        input_data: Input data to convert
        return_as_base64: If True, returns base64 encoded strings instead of PIL Images
        max_edge_size: If specified, images will be resized so that their largest edge
                      is at most this many pixels, maintaining aspect ratio

    Returns:
        Dictionary with keys formatted as "filename_XXX" where XXX is the page number
        with leading zeros. For single images, the key will be "filename_001".
    """
    if input_data is None:
        return {}

    if isinstance(input_data, Iterable) and not isinstance(
        input_data, str | bytes | Path
    ):
        result = {}
        for item in input_data:
            # Use a tuple of (item, max_edge_size, return_as_base64) as part of the cache key for each item
            result.update(sanitize_to_images(item, return_as_base64, max_edge_size))
        return result

    # Handle folder input
    if isinstance(input_data, str | Path):
        path = Path(input_data)
        if path.is_dir():
            result = {}
            # Supported file extensions
            supported_extensions = {
                ".pdf",
                ".png",
                ".jpg",
                ".jpeg",
                ".gif",
                ".bmp",
                ".tiff",
                ".tif",
                ".webp",
            }

            for file_path in path.iterdir():
                if (
                    file_path.is_file()
                    and file_path.suffix.lower() in supported_extensions
                ):
                    file_result = sanitize_to_images(
                        file_path, return_as_base64, max_edge_size
                    )
                    # rename keys that already exist in result:
                    modified_keys = []
                    for key, value in file_result.items():
                        if key in result:
                            result[f"{key}_001"] = value
                            modified_keys.append(key)
                        else:
                            result[key] = value

                    # remove keys that were modified
                    for key in modified_keys:
                        del result[key]
                    result.update(file_result)
            return result

    cache_manager = CacheManager()

    # Compute a unique hash for the input and processing parameters
    if isinstance(input_data, str | Path):
        input_hash = cache_manager.compute_file_hash(Path(input_data))
        base_name = Path(input_data).stem
    elif isinstance(input_data, bytes):
        input_hash = cache_manager.compute_bytes_hash(input_data)
        base_name = "image"
    elif isinstance(input_data, Image.Image):
        # For PIL Images, use their bytes as hash
        img_bytes = BytesIO()
        input_data.save(img_bytes, format="PNG")
        input_hash = cache_manager.compute_bytes_hash(img_bytes.getvalue())
        base_name = "image"
    else:
        raise ValueError(f"Unsupported input type: {type(input_data)}")

    cache_key = f"{input_hash}_size{max_edge_size}_b64{return_as_base64}"
    cached = cache_manager.get(cache_key)
    if cached:
        # Deserialize the cached result
        result = pickle.loads(cached[0])
        return result

    pdf_converter = PDFConverter(cache_manager)
    image_loader = ImageLoader(pdf_converter, max_edge_size)

    # Choose appropriate loader based on return type, make the annotation explicit for mypy
    loader: ImageLoader | Base64ImageLoader
    if return_as_base64:
        loader = Base64ImageLoader(image_loader)
    else:
        loader = image_loader

    try:
        images = loader.load(input_data)
        result = {}

        # Handle single image case
        if len(images) == 1:
            result[f"{base_name}_001"] = images[0]
        else:
            # Handle multiple images (e.g., from PDF)
            for i, img in enumerate(images, 1):
                result[f"{base_name}_{i:03d}"] = img

        # Store the final result in the cache
        cache_manager.set(cache_key, [pickle.dumps(result)])
        cache_manager.save()
        return result
    except Exception as e:
        cache_manager.save()
        raise e


def save_sanitized_images(
    input_data: str | Path | bytes | Image.Image | Iterable | None,
    destination: Path | None = None,
    max_edge_size: int | None = None,
) -> Path:
    """Convert input data to images and save them to the specified destination.

    Args:
        input_data: Input data to convert (PDF, image, folder, etc.)
        destination: Destination folder for saved images. If None, uses "single_pages"
        max_edge_size: If specified, images will be resized so that their largest edge
                      is at most this many pixels, maintaining aspect ratio

    Returns:
        Path to the destination folder where images were saved
    """
    if destination is None:
        destination = Path("single_pages")

    destination.mkdir(parents=True, exist_ok=True)

    # Get sanitized images
    images_dict = sanitize_to_images(
        input_data, return_as_base64=False, max_edge_size=max_edge_size
    )

    # Save each image
    for key, image in images_dict.items():
        if isinstance(image, Image.Image):
            # Add .png extension if not present
            filename = (
                f"{key}.png"
                if not key.endswith(
                    (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".tif", ".webp")
                )
                else key
            )
            output_path = destination / filename
            image.save(output_path, "PNG")
            print(f"Saved: {output_path}")

    print(f"Saved {len(images_dict)} images to {destination}")
    return destination
