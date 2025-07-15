"""Tests for the image data loader functionality."""

from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

import pytest
from PIL import Image
from io import BytesIO

from bubbola.image_data_loader import PDFConverter, CacheManager, sanitize_to_images


class TestPDFConverter:
    """Test cases for PDFConverter."""

    def test_pdf_converter_initialization(self):
        """Test that PDFConverter initializes correctly."""
        cache_manager = CacheManager()
        converter = PDFConverter(cache_manager)
        assert converter.cache_manager is cache_manager

    @patch('bubbola.image_data_loader.fitz')
    def test_convert_from_path_calls_pymupdf(self, mock_fitz):
        """Test that convert_from_path uses PyMuPDF correctly."""
        # Mock the fitz document and page
        mock_doc = MagicMock()
        mock_page = Mock()
        mock_pixmap = Mock()
        
        mock_doc.__len__.return_value = 1
        mock_doc.load_page.return_value = mock_page
        mock_fitz.open.return_value = mock_doc
        mock_fitz.Matrix.return_value = Mock()
        mock_page.get_pixmap.return_value = mock_pixmap
        
        # Create a mock PNG image
        mock_img = Image.new('RGB', (100, 100), color='red')
        img_bytes_io = BytesIO()
        mock_img.save(img_bytes_io, format='PNG')
        img_bytes = img_bytes_io.getvalue()
        mock_pixmap.tobytes.return_value = img_bytes
        
        cache_manager = CacheManager()
        converter = PDFConverter(cache_manager)
        
        # Test conversion
        test_path = Path("test.pdf")
        result = converter.convert_from_path(test_path)
        
        # Verify PyMuPDF was called correctly
        mock_fitz.open.assert_called_once_with(str(test_path))
        mock_doc.load_page.assert_called_once_with(0)
        mock_doc.close.assert_called_once()
        
        # Verify result is a list of PIL Images
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], Image.Image)

    @patch('bubbola.image_data_loader.fitz')
    def test_convert_from_bytes_calls_pymupdf(self, mock_fitz):
        """Test that convert_from_bytes uses PyMuPDF correctly."""
        # Mock the fitz document and page
        mock_doc = MagicMock()
        mock_page = Mock()
        mock_pixmap = Mock()
        
        mock_doc.__len__.return_value = 1
        mock_doc.load_page.return_value = mock_page
        mock_fitz.open.return_value = mock_doc
        mock_fitz.Matrix.return_value = Mock()
        mock_page.get_pixmap.return_value = mock_pixmap
        
        # Create a mock PNG image
        mock_img = Image.new('RGB', (100, 100), color='blue')
        img_bytes_io = BytesIO()
        mock_img.save(img_bytes_io, format='PNG')
        img_bytes = img_bytes_io.getvalue()
        mock_pixmap.tobytes.return_value = img_bytes
        
        cache_manager = CacheManager()
        converter = PDFConverter(cache_manager)
        
        # Test conversion
        test_bytes = b"fake pdf content"
        result = converter.convert_from_bytes(test_bytes)
        
        # Verify PyMuPDF was called correctly
        mock_fitz.open.assert_called_once_with(stream=test_bytes, filetype="pdf")
        mock_doc.load_page.assert_called_once_with(0)
        mock_doc.close.assert_called_once()
        
        # Verify result is a list of PIL Images
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], Image.Image)


class TestSanitizeToImages:
    """Test cases for sanitize_to_images function."""

    def test_sanitize_to_images_with_none(self):
        """Test sanitize_to_images with None input."""
        result = sanitize_to_images(None)
        assert result == {}

    def test_sanitize_to_images_with_pil_image(self):
        """Test sanitize_to_images with PIL Image input."""
        test_image = Image.new('RGB', (100, 100), color='green')
        result = sanitize_to_images(test_image)
        
        assert len(result) == 1
        assert "image_001" in result
        assert isinstance(result["image_001"], Image.Image)
        assert result["image_001"].size == (100, 100)

    def test_sanitize_to_images_with_iterable(self):
        """Test sanitize_to_images with iterable input."""
        test_images = [
            Image.new('RGB', (50, 50), color='red'),
            Image.new('RGB', (75, 75), color='blue')
        ]
        result = sanitize_to_images(test_images)
        
        # Check that the result contains two unique images
        assert len(result) == 2
        unique_sizes = {img.size for img in result.values() if isinstance(img, Image.Image)}
        assert (50, 50) in unique_sizes
        assert (75, 75) in unique_sizes

    def test_sanitize_to_images_with_resizing(self):
        """Test sanitize_to_images with max_edge_size parameter."""
        test_image = Image.new('RGB', (200, 100), color='yellow')
        result = sanitize_to_images(test_image, max_edge_size=50)
        
        assert len(result) == 1
        assert "image_001" in result
        resized_image = result["image_001"]
        assert resized_image.size == (50, 25)  # Maintains aspect ratio

    def test_sanitize_to_images_with_base64_output(self):
        """Test sanitize_to_images with return_as_base64=True."""
        test_image = Image.new('RGB', (50, 50), color='purple')
        result = sanitize_to_images(test_image, return_as_base64=True)
        
        assert len(result) == 1
        assert "image_001" in result
        assert isinstance(result["image_001"], str)
        # Base64 strings should start with data:image/png;base64, or be pure base64
        assert len(result["image_001"]) > 0 