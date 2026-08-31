from __future__ import annotations

import zipfile
from pathlib import Path

import pytest
from fastapi import HTTPException

from api.main import _safe_file_name, _validate_file_signature


@pytest.mark.parametrize("name", ["../secret.txt", "folder/file.pdf", "..\\secret.txt", "x.exe", ""])
def test_unsafe_or_unsupported_file_names_are_rejected(name: str) -> None:
    with pytest.raises(HTTPException):
        _safe_file_name(name)


def test_file_signature_validation(tmp_path: Path) -> None:
    fake_pdf = tmp_path / "fake.pdf"
    fake_pdf.write_bytes(b"not pdf")
    with pytest.raises(HTTPException):
        _validate_file_signature(fake_pdf, ".pdf")

    pdf = tmp_path / "ok.pdf"
    pdf.write_bytes(b"%PDF-1.7\n")
    _validate_file_signature(pdf, ".pdf")

    fake_docx = tmp_path / "fake.docx"
    with zipfile.ZipFile(fake_docx, "w") as archive:
        archive.writestr("xl/workbook.xml", "x")
    with pytest.raises(HTTPException):
        _validate_file_signature(fake_docx, ".docx")


def test_new_format_names_signatures_and_archive_traversal(tmp_path: Path) -> None:
    for name in ("deck.pptx", "page.html", "data.json", "events.jsonl", "safe.xml"):
        assert _safe_file_name(name) == name

    pptx = tmp_path / "deck.pptx"
    with zipfile.ZipFile(pptx, "w") as archive:
        archive.writestr("ppt/presentation.xml", "x")
    _validate_file_signature(pptx, ".pptx")

    malicious = tmp_path / "malicious.pptx"
    with zipfile.ZipFile(malicious, "w") as archive:
        archive.writestr("ppt/presentation.xml", "x")
        archive.writestr("../outside.txt", "escape")
    with pytest.raises(HTTPException, match="unsafe archive member"):
        _validate_file_signature(malicious, ".pptx")


@pytest.mark.parametrize(
    ("name", "extension"),
    [
        ("broken.pdf", ".pdf"),
        ("broken.docx", ".docx"),
        ("broken.xlsx", ".xlsx"),
        ("broken.csv", ".csv"),
        ("broken.png", ".png"),
        ("broken.txt", ".txt"),
        ("broken.md", ".md"),
        ("broken.pptx", ".pptx"),
        ("broken.html", ".html"),
        ("broken.json", ".json"),
        ("broken.xml", ".xml"),
    ],
)
def test_corrupt_payload_is_rejected_for_each_format_family(
    tmp_path: Path, name: str, extension: str
) -> None:
    path = tmp_path / name
    path.write_bytes(b"\x00corrupt payload")
    with pytest.raises(HTTPException):
        _validate_file_signature(path, extension)


@pytest.mark.parametrize(
    "extension",
    ["pdf", "docx", "xlsx", "csv", "png", "txt", "md", "pptx", "html", "json", "xml"],
)
def test_directory_traversal_is_rejected_for_each_format_family(extension: str) -> None:
    with pytest.raises(HTTPException, match="unsafe file name"):
        _safe_file_name(f"../payload.{extension}")
