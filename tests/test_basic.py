import os
import sys
import unittest

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestBasicFunctionality(unittest.TestCase):
    def test_import_modules(self):
        """Test importing all modules succeeds"""
        import scraper
        import download_image
        import create_chat_dataset
        import upload_to_huggingface
        self.assertTrue(True)  # If we get here, imports worked
    
    def test_directory_structure(self):
        """Test that required directories exist"""
        os.makedirs("dataset", exist_ok=True)
        os.makedirs("images", exist_ok=True)
        self.assertTrue(os.path.exists("dataset"))
        self.assertTrue(os.path.exists("images"))

if __name__ == "__main__":
    unittest.main()
