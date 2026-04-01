import os
import tempfile
from pathlib import Path
import pytest

from doc_mcp import server


def test_resolve_within_root_allows_relative_and_rejects_escape(tmp_path):
    root = tmp_path / "docs"
    root.mkdir()
    # create a file inside
    f = root / "a.txt"
    f.write_text("hello")

    # relative path should resolve
    resolved = server._resolve_within_root(root, "a.txt")
    assert resolved.exists()
    assert resolved == f.resolve()

    # try to escape using ..
    outside = tmp_path / "outside.txt"
    outside.write_text("nope")
    with pytest.raises(PermissionError):
        server._resolve_within_root(root, "../outside.txt")


def test_list_and_read_files_use_relative_paths(tmp_path):
    root = tmp_path / "docs"
    root.mkdir()
    sub = root / "sub"
    sub.mkdir()
    f1 = root / "one.md"
    f1.write_text("content one")
    f2 = sub / "two.txt"
    f2.write_text("content two")

    svc = server.GameDocServer(docs_root=str(root))

    files = svc.list_files("")
    # Should return relative paths
    assert "one.md" in files
    assert str(Path("sub") / "two.txt") in files

    content = svc.read_document("one.md")
    assert "content one" in content

    # Absolute path referencing inside root should be allowed
    content2 = svc.read_document(str(f2))
    assert "content two" in content2

    # Request outside the root should fail
    outside = tmp_path / "outside.md"
    outside.write_text("x")
    with pytest.raises(PermissionError):
        svc.read_document(str(outside))