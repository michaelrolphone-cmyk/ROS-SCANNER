import os
import unittest
from pathlib import Path
from unittest.mock import patch

import survey_recon as sr


class DummyImage:
    size = (100, 80)


class OcrEngineTests(unittest.TestCase):
    def test_detect_default_ocr_engine_apple_silicon(self):
        with patch("survey_recon.platform.system", return_value="Darwin"), patch(
            "survey_recon.platform.machine", return_value="arm64"
        ):
            self.assertEqual(sr.detect_default_ocr_engine(), "mineru")

    def test_resolve_ocr_engine_env_override(self):
        with patch.dict(os.environ, {"SURVEY_RECON_OCR_ENGINE": "tesseract"}, clear=False):
            self.assertEqual(sr.resolve_ocr_engine(), "tesseract")

    def test_check_ocr_ready_mineru_missing(self):
        with patch("survey_recon.resolve_ocr_engine", return_value="mineru"), patch(
            "survey_recon.shutil.which", return_value=None
        ):
            ok, msg = sr.check_ocr_ready()
        self.assertFalse(ok)
        self.assertIn("mineru_missing", msg)

    def test_ocr_scan_lines_uses_mineru_lines(self):
        with patch("survey_recon.resolve_ocr_engine", return_value="mineru"), patch(
            "survey_recon._mineru_image_to_lines", return_value=["N 10 E 20.0", "S 20 W 15.0"]
        ):
            lines = sr.ocr_scan_lines(DummyImage(), image_path=Path("fake.png"))
        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0].text, "N 10 E 20.0")


if __name__ == "__main__":
    unittest.main()
