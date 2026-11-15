"""The clean-up for browser copies left behind before 2026-11-15."""

from __future__ import annotations

import os
import time

from tools.delete_orphan_web_videos import orphan_copies


def test_only_old_copies_whose_original_is_gone_are_listed(tmp_path):
    old = time.time() - 48 * 3600
    files = {
        "gone1_web.mp4": old,        # original deleted: listed
        "kept2.mov": old,            # an original (any extension) ...
        "kept2_web.mp4": old,        # ... so its copy stays
        "fresh3_web.mp4": time.time(),   # too new: an upload may be in progress
        "notes_web.txt": old,        # not a video copy
        "_web.mp4": old,             # no job name
    }
    for name, moment in files.items():
        path = tmp_path / name
        path.write_bytes(b"x")
        os.utime(path, (moment, moment))
    (tmp_path / "gone4_web.mp4").mkdir()     # a folder, not a file

    assert [path.name for path in orphan_copies(tmp_path, 24)] == ["gone1_web.mp4"]
    assert orphan_copies(tmp_path / "missing", 24) == []
