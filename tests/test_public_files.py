from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pytest import MonkeyPatch

from app.config import config
from main import app


@pytest.mark.parametrize(
    "relative_path",
    ["analytics.log", "about_me.md", "vector_db/index.bin", "CV_Demo.pdf"],
)
def test_data_files_are_not_publicly_served(
    monkeypatch: MonkeyPatch, tmp_path: Path, relative_path: str
) -> None:
    # Existing files are required: a 404 for a missing file would prove nothing.
    data_file = tmp_path / "data" / relative_path
    data_file.parent.mkdir(parents=True, exist_ok=True)
    private_content = b"private synthetic test data"
    data_file.write_bytes(private_content)
    monkeypatch.chdir(tmp_path)

    with TestClient(app) as client:
        response = client.get(f"/data/{relative_path}")

    assert response.status_code == 404
    assert private_content not in response.content


def test_resume_download_serves_only_the_configured_pdf(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    resume = tmp_path / "candidate.pdf"
    contents = b"%PDF-1.4\nsynthetic resume fixture\n%%EOF\n"
    resume.write_bytes(contents)
    monkeypatch.setattr(config.data, "cv_path", str(resume))

    with TestClient(app) as client:
        response = client.get("/resume")

    assert response.status_code == 200
    assert response.content == contents
    assert response.headers["content-type"] == "application/pdf"
    assert response.headers["content-disposition"] == 'attachment; filename="CV.pdf"'


def test_resume_query_cannot_select_a_different_file(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    resume = tmp_path / "candidate.pdf"
    resume.write_bytes(b"public synthetic resume")
    private_file = tmp_path / "analytics.log"
    private_file.write_bytes(b"private synthetic analytics")
    monkeypatch.setattr(config.data, "cv_path", str(resume))

    with TestClient(app) as client:
        response = client.get("/resume", params={"path": str(private_file)})

    assert response.status_code == 200
    assert response.content == resume.read_bytes()
    assert private_file.read_bytes() not in response.content


@pytest.mark.parametrize("is_directory", [False, True])
def test_unavailable_resume_returns_404(
    monkeypatch: MonkeyPatch, tmp_path: Path, is_directory: bool
) -> None:
    resume = tmp_path / "unavailable.pdf"
    if is_directory:
        resume.mkdir()
    monkeypatch.setattr(config.data, "cv_path", str(resume))

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/resume")

    assert response.status_code == 404
    assert response.json() == {"detail": "Resume not found"}
    assert str(resume) not in response.text


def test_homepage_links_to_resume_not_to_filesystem_path(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    resume = tmp_path / "private-directory" / "candidate.pdf"
    monkeypatch.setattr(config.data, "cv_path", str(resume))

    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert 'href="/resume"' in response.text
    assert str(resume) not in response.text
