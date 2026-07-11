import os
import pytest
import fitz
from backend.app.services.parser_service import ParserService, PDFParser

@pytest.fixture
def sample_pdf(tmp_path):
    pdf_path = os.path.join(tmp_path, "test_doc.pdf")
    doc = fitz.open()
    
    # Page 1
    page1 = doc.new_page()
    page1.insert_text((50, 50), "This is page one text content.")
    
    # Page 2
    page2 = doc.new_page()
    page2.insert_text((55, 55), "Here is page two data.")
    
    doc.save(pdf_path)
    doc.close()
    
    yield pdf_path
    
    if os.path.exists(pdf_path):
        try:
            os.remove(pdf_path)
        except Exception:
            pass

def test_pdf_parser_success(sample_pdf):
    parser = PDFParser()
    pages = parser.parse(sample_pdf)
    
    assert len(pages) == 2
    assert pages[0]["page"] == 1
    assert "page one text content" in pages[0]["text"]
    assert pages[1]["page"] == 2
    assert "page two data" in pages[1]["text"]

def test_parser_service_pdf_routing(sample_pdf):
    service = ParserService()
    # Test routing by file extension in file path
    pages = service.parse_file(sample_pdf, "application/octet-stream")
    assert len(pages) == 2
    assert "page one text content" in pages[0]["text"]

    # Test routing by content-type
    pages = service.parse_file(sample_pdf, "application/pdf")
    assert len(pages) == 2
    assert "page two data" in pages[1]["text"]

def test_parser_service_unsupported():
    service = ParserService()
    with pytest.raises(ValueError) as excinfo:
        service.parse_file("text.docx", "application/vnd.openxmlformats")
    assert "Unsupported content type" in str(excinfo.value)
