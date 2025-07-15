"""Tests for the image data loader functionality."""

import shutil
import tempfile
from pathlib import Path

import pytest
from PIL import Image

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
        return [assets_dir / "0088_001.pdf", assets_dir / "0089_001.pdf"]

    @pytest.fixture
    def single_page_files(self, assets_dir):
        """Get the extracted single page PNG files."""
        return [
            assets_dir / "single_pages" / "0088_001_001.png",
            assets_dir / "single_pages" / "0089_001_001.png",
            assets_dir / "single_pages" / "0089_001_002.png",
        ]

    @pytest.fixture
    def resized_files(self, assets_dir):
        """Get the resized PNG files."""
        return [
            assets_dir / "single_pages_resized" / "0088_001_001.png",
            assets_dir / "single_pages_resized" / "0089_001_001.png",
            assets_dir / "single_pages_resized" / "0089_001_002.png",
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

    def test_folder_processing_with_pdfs_and_images(self, assets_dir):
        """Test processing a folder containing PDFs and images."""
        # Create a temporary test folder with mixed content
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Copy PDF files
            pdf1 = temp_path / "test1.pdf"
            pdf2 = temp_path / "test2.pdf"
            shutil.copy2(assets_dir / "0088_001.pdf", pdf1)
            shutil.copy2(assets_dir / "0089_001.pdf", pdf2)

            # Copy image files with different names to avoid conflicts
            img1 = temp_path / "image1.png"
            img2 = temp_path / "image2.jpg"
            shutil.copy2(assets_dir / "single_pages" / "0088_001_001.png", img1)
            shutil.copy2(assets_dir / "single_pages" / "0089_001_001.png", img2)

            # Create an unsupported file (should be ignored)
            unsupported = temp_path / "test.txt"
            unsupported.write_text("This should be ignored")

            # Process the folder
            result = sanitize_to_images(temp_path)

            # Should have 5 total pages: 1 from test1.pdf + 2 from test2.pdf + 1 from image1.png + 1 from image2.jpg
            assert len(result) == 5
            assert "test1_001" in result  # PDF page
            assert "test2_001" in result  # PDF page 1
            assert "test2_002" in result  # PDF page 2
            assert "image1_001" in result  # PNG image
            assert "image2_001" in result  # JPG image

            # All results should be PIL Images
            assert all(isinstance(img, Image.Image) for img in result.values())

    def test_folder_processing_with_resizing(self, assets_dir):
        """Test processing a folder with image resizing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Copy files
            pdf_file = temp_path / "test.pdf"
            img_file = temp_path / "test.png"
            shutil.copy2(assets_dir / "0088_001.pdf", pdf_file)
            shutil.copy2(assets_dir / "single_pages" / "0088_001_001.png", img_file)

            # Process with resizing
            result = sanitize_to_images(temp_path, max_edge_size=100)

            assert len(result) == 2  # 1 PDF page + 1 image
            for img in result.values():
                width, height = img.size
                assert max(width, height) <= 100
                assert min(width, height) > 0

    def test_folder_processing_with_base64(self, assets_dir):
        """Test processing a folder with base64 output."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Copy files
            pdf_file = temp_path / "test.pdf"
            img_file = temp_path / "test.png"
            shutil.copy2(assets_dir / "0088_001.pdf", pdf_file)
            shutil.copy2(assets_dir / "single_pages" / "0088_001_001.png", img_file)

            # Process with base64 output
            result = sanitize_to_images(temp_path, return_as_base64=True)

            assert len(result) == 2
            assert all(isinstance(img, str) for img in result.values())
            assert all(len(img) > 0 for img in result.values())

    def test_folder_processing_empty_folder(self):
        """Test processing an empty folder."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            result = sanitize_to_images(temp_path)
            assert result == {}

    def test_folder_processing_only_unsupported_files(self):
        """Test processing a folder with only unsupported file types."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create unsupported files
            (temp_path / "test1.txt").write_text("Text file")
            (temp_path / "test2.doc").write_text("Document file")
            (temp_path / "test3.py").write_text("Python file")

            result = sanitize_to_images(temp_path)
            assert result == {}

    def test_folder_processing_case_insensitive_extensions(self, assets_dir):
        """Test that folder processing handles case-insensitive file extensions."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Copy files with different case extensions but different base names
            pdf_upper = temp_path / "test1.PDF"
            img_upper = temp_path / "test2.PNG"
            img_mixed = temp_path / "test3.JpG"

            shutil.copy2(assets_dir / "0088_001.pdf", pdf_upper)
            shutil.copy2(assets_dir / "single_pages" / "0088_001_001.png", img_upper)
            shutil.copy2(assets_dir / "single_pages" / "0089_001_001.png", img_mixed)

            result = sanitize_to_images(temp_path)

            # Should process all files regardless of case
            assert len(result) == 3
            assert "test1_001" in result  # PDF
            assert "test2_001" in result  # PNG
            assert "test3_001" in result  # JPG

    def test_folder_processing_string_path(self, assets_dir):
        """Test processing a folder using string path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Copy a file
            pdf_file = temp_path / "test.pdf"
            shutil.copy2(assets_dir / "0088_001.pdf", pdf_file)

            # Process using string path
            result = sanitize_to_images(str(temp_path))

            assert len(result) == 1
            assert "test_001" in result
            assert isinstance(result["test_001"], Image.Image)

    def test_folder_processing_nested_structure(self, assets_dir):
        """Test that folder processing only processes files in the root folder, not subdirectories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create nested structure
            subdir = temp_path / "subdir"
            subdir.mkdir()

            # Files in root directory (should be processed)
            root_pdf = temp_path / "root1.pdf"
            root_img = temp_path / "root2.png"
            shutil.copy2(assets_dir / "0088_001.pdf", root_pdf)
            shutil.copy2(assets_dir / "single_pages" / "0088_001_001.png", root_img)

            # Files in subdirectory (should NOT be processed)
            sub_pdf = subdir / "sub.pdf"
            sub_img = subdir / "sub.png"
            shutil.copy2(assets_dir / "0089_001.pdf", sub_pdf)
            shutil.copy2(assets_dir / "single_pages" / "0089_001_001.png", sub_img)

            result = sanitize_to_images(temp_path)

            # Should only process files in root directory
            assert len(result) == 2
            assert "root1_001" in result  # PDF
            assert "root2_001" in result  # PNG

            # Should NOT include subdirectory files
            assert "sub_001" not in result
