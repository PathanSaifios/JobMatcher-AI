import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import pandas as pd
from src.data.preprocessing import clean_text

def test_clean_text_removes_urls():
    raw = "Visit https://example.com for details."
    assert "https" not in clean_text(raw)

def test_clean_text_lowercase():
    assert clean_text("PYTHON DEVELOPER") == "python developer"

def test_clean_text_extra_whitespace():
    result = clean_text("  hello   world  ")
    assert "  " not in result
    assert result.strip() == result

def test_clean_text_empty_string():
    assert clean_text("") == ""

def test_clean_text_non_string():
    assert clean_text(None) == ""
    assert clean_text(123) == ""
