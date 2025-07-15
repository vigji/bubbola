"""Tests for the image data loader functionality."""

from pathlib import Path
import pytest
from PIL import Image
import tempfile
import shutil
from unittest import mock

from bubbola.image_data_loader import sanitize_to_images


class TestImageDataLoader:
    """Test cases for image data loader using real assets."""

    @pytest.fixture(autouse=True)
    def isolated_cache(self, monkeypatch):
        """Patch CacheManager in the image_data_loader module to use a temp directory for cache during tests."""
        temp_dir = tempfile.mkdtemp()
        from bubbola import image_data_loader
        original_cache_manager = image_data_loader.CacheManager
        
        class TempCacheManager(original_cache_manager):
            def __init__(self, cache_dir=None, max_age_days=30):
                super().__init__(cache_dir=Path(temp_dir), max_age_days=max_age_days)
        
        monkeypatch.setattr(image_data_loader, "CacheManager", TempCacheManager)
        yield
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def assets_dir(self):
        """Get the assets directory path."""
        return Path(__file__).parent / "assets"

    @pytest.fixture
    def pdf_files(self, assets_dir):
        """Get the test PDF files."""
        return [
            assets_dir / "0088_001.pdf",
            assets_dir / "0089_001.pdf"
        ]

    @pytest.fixture
    def single_page_files(self, assets_dir):
        """Get the extracted single page PNG files."""
        return [
            assets_dir / "single_pages" / "0088_001_001.png",
            assets_dir / "single_pages" / "0089_001_001.png",
            assets_dir / "single_pages" / "0089_001_002.png"
        ]

    @pytest.fixture
    def resized_files(self, assets_dir):
        """Get the resized PNG files."""
        return [
            assets_dir / "single_pages_resized" / "0088_001_001.png",
            assets_dir / "single_pages_resized" / "0089_001_001.png",
            assets_dir / "single_pages_resized" / "0089_001_002.png"
        ]

    def test_load_single_png_file(self, single_page_files):
        """Test loading a single PNG file."""
        png_file = single_page_files[0]
        result = sanitize_to_images(png_file)
        
        assert len(result) == 1
        assert "0088_001_001_001" in result
        assert isinstance(result["0088_001_001_001"], Image.Image)
        # Verify the loaded image has reasonable dimensions
        assert result["0088_001_001_001"].size[0] > 0
        assert result["0088_001_001_001"].size[1] > 0

    def test_load_single_pdf_file(self, pdf_files):
        """Test loading a single PDF file."""
        pdf_file = pdf_files[0]
        result = sanitize_to_images(pdf_file)
        
        assert len(result) == 1
        assert "0088_001_001" in result
        assert isinstance(result["0088_001_001"], Image.Image)
        # Verify the loaded image has reasonable dimensions
        assert result["0088_001_001"].size[0] > 0
        assert result["0088_001_001"].size[1] > 0

    def test_load_multi_page_pdf(self, pdf_files):
        """Test loading a multi-page PDF file."""
        pdf_file = pdf_files[1]  # 0089_001.pdf has 2 pages
        result = sanitize_to_images(pdf_file)
        
        assert len(result) == 2
        assert "0089_001_001" in result
        assert "0089_001_002" in result
        assert isinstance(result["0089_001_001"], Image.Image)
        assert isinstance(result["0089_001_002"], Image.Image)

    def test_load_pdf_as_string_path(self, pdf_files):
        """Test loading PDF using string path."""
        pdf_file = str(pdf_files[0])
        result = sanitize_to_images(pdf_file)
        
        assert len(result) == 1
        assert "0088_001_001" in result
        assert isinstance(result["0088_001_001"], Image.Image)

    def test_load_list_of_pdf_files(self, pdf_files):
        """Test loading a list of PDF files."""
        result = sanitize_to_images(pdf_files)
        
        # Should have 3 total pages: 1 from 0088_001.pdf + 2 from 0089_001.pdf
        assert len(result) == 3
        assert "0088_001_001" in result
        assert "0089_001_001" in result
        assert "0089_001_002" in result
        assert all(isinstance(img, Image.Image) for img in result.values())

    def test_load_list_of_pdf_strings(self, pdf_files):
        """Test loading a list of PDF files as strings."""
        pdf_strings = [str(pdf) for pdf in pdf_files]
        result = sanitize_to_images(pdf_strings)
        
        assert len(result) == 3
        assert "0088_001_001" in result
        assert "0089_001_001" in result
        assert "0089_001_002" in result

    def test_resize_single_image(self, single_page_files):
        """Test resizing a single image."""
        png_file = single_page_files[0]
        result = sanitize_to_images(png_file, max_edge_size=100)
        
        assert len(result) == 1
        assert "0088_001_001_001" in result
        resized_image = result["0088_001_001_001"]
        
        # Verify resizing maintains aspect ratio and respects max_edge_size
        width, height = resized_image.size
        assert max(width, height) <= 100
        assert min(width, height) > 0

    def test_resize_pdf_pages(self, pdf_files):
        """Test resizing PDF pages."""
        pdf_file = pdf_files[1]  # Multi-page PDF
        result = sanitize_to_images(pdf_file, max_edge_size=100)
        
        assert len(result) == 2
        for img in result.values():
            width, height = img.size
            assert max(width, height) <= 100
            assert min(width, height) > 0

    def test_resize_list_of_files(self, pdf_files):
        """Test resizing a list of files."""
        result = sanitize_to_images(pdf_files, max_edge_size=100)
        
        assert len(result) == 3
        for img in result.values():
            width, height = img.size
            assert max(width, height) <= 100
            assert min(width, height) > 0

    def test_base64_output(self, single_page_files):
        """Test returning images as base64 strings."""
        png_file = single_page_files[0]
        result = sanitize_to_images(png_file, return_as_base64=True)
        
        assert len(result) == 1
        assert "0088_001_001_001" in result
        assert isinstance(result["0088_001_001_001"], str)
        # Base64 strings should be non-empty
        assert len(result["0088_001_001_001"]) > 0

    def test_base64_with_resizing(self, pdf_files):
        """Test base64 output with resizing."""
        pdf_file = pdf_files[0]
        result = sanitize_to_images(pdf_file, max_edge_size=100, return_as_base64=True)
        
        assert len(result) == 1
        assert "0088_001_001" in result
        assert isinstance(result["0088_001_001"], str)
        assert len(result["0088_001_001"]) > 0

    def test_none_input(self):
        """Test handling of None input."""
        result = sanitize_to_images(None)
        assert result == {}

    def test_empty_list_input(self):
        """Test handling of empty list input."""
        result = sanitize_to_images([])
        assert result == {}

    def test_file_not_found(self):
        """Test handling of non-existent file."""
        with pytest.raises(FileNotFoundError):
            sanitize_to_images("nonexistent_file.pdf")

    def test_unsupported_input_type(self):
        """Test handling of unsupported input type."""
        with pytest.raises(ValueError, match="Unsupported input type"):
            sanitize_to_images(123)  # Integer is not supported 