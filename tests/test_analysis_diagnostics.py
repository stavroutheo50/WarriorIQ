"""What an analysis records about the machine before it starts.

A run thirty times slower than the same code on the same hardware looks, from
outside, exactly like a run that is simply heavy. An afternoon went into
ruling out the code for one of those, and the answer was never in the code.
Two numbers at the start - free VRAM and free system memory - make the next
one self-explaining.
"""
from __future__ import annotations

import logging
import unittest
from unittest import mock

from core import analyzer
from core.config import SETTINGS


class ClipBufferTests(unittest.TestCase):
    """SAM2's frame buffer is ordinary RAM, and it is the big number."""

    def test_the_buffer_tracks_the_configured_chunk(self):
        previous = SETTINGS.sam_continuous_chunk_frames
        try:
            object.__setattr__(SETTINGS, "sam_continuous_chunk_frames", 120)
            small = analyzer._sam_clip_buffer_bytes()
            object.__setattr__(SETTINGS, "sam_continuous_chunk_frames", 360)
            large = analyzer._sam_clip_buffer_bytes()
        finally:
            object.__setattr__(SETTINGS, "sam_continuous_chunk_frames", previous)
        self.assertEqual(large, small * 3)

    def test_the_default_chunk_keeps_the_pass_in_one_piece(self):
        """The chunk must still cover a whole SAM2 pass.

        Lowering it would cut the buffer - 120 frames is 1.41 GB against
        4.22 GB - but SAM2 re-seeds at every chunk boundary, and measured over
        263 frames that moved the median box by 4px. One chunk is the
        reproducible path, so the memory problem is reported rather than
        traded away silently. See core/config.py.
        """
        from core.config import SETTINGS
        self.assertGreaterEqual(SETTINGS.sam_continuous_chunk_frames,
                                SETTINGS.sam_continuous_max_frames)

    def test_the_arithmetic_matches_what_sam2_allocates(self):
        """capacity x 3 x 1024 x 1024 float32, as _DecodedClip declares it."""
        frames = int(SETTINGS.sam_continuous_chunk_frames)
        self.assertEqual(analyzer._sam_clip_buffer_bytes(), frames * 3 * 1024 * 1024 * 4)


class HostMemoryLoggingTests(unittest.TestCase):
    def test_plenty_of_memory_logs_without_warning(self):
        plenty = mock.Mock(available=32 * 1024 ** 3, total=64 * 1024 ** 3)
        with mock.patch.dict("sys.modules", {"psutil": mock.Mock(
                virtual_memory=mock.Mock(return_value=plenty))}):
            with self.assertLogs("warrioriq.analysis", level="INFO") as logs:
                analyzer._log_host_memory()
        joined = " ".join(logs.output)
        self.assertIn("analysis_host_memory", joined)
        self.assertNotIn("analysis_host_memory_tight", joined)

    def test_a_healthy_run_is_not_warned_about(self):
        """The exact false positive this threshold was rebuilt to remove.

        A real analysis started with 7.08 GB available against a 4.22 GB
        buffer, warned, and then finished a 94-second clip in 103 seconds at
        full speed. A warning that cries wolf on a healthy run teaches the
        reader to ignore the one that matters.
        """
        healthy = mock.Mock(available=7.08 * 1024 ** 3, total=15.87 * 1024 ** 3)
        with mock.patch.dict("sys.modules", {"psutil": mock.Mock(
                virtual_memory=mock.Mock(return_value=healthy))}):
            with self.assertLogs("warrioriq.analysis", level="INFO") as logs:
                analyzer._log_host_memory()
        self.assertNotIn("analysis_host_memory_tight", " ".join(logs.output))

    def test_the_threshold_is_the_buffer_plus_a_measured_working_set(self):
        """Not a multiple of the buffer.

        Sampling RSS through a real pass: 5.33 GB peak against a 4.22 GB
        buffer, so 1.11 GB beside it - the pose model and its TensorRT
        context, SAM2's weights, torch and the decode working set.
        """
        self.assertGreaterEqual(analyzer.ANALYSIS_WORKING_SET_GB, 1.1)
        self.assertLess(analyzer.ANALYSIS_WORKING_SET_GB, 3.0)

    def test_a_tight_machine_is_warned_about(self):
        tight = mock.Mock(available=1 * 1024 ** 3, total=16 * 1024 ** 3)
        with mock.patch.dict("sys.modules", {"psutil": mock.Mock(
                virtual_memory=mock.Mock(return_value=tight))}):
            with self.assertLogs("warrioriq.analysis", level="WARNING") as logs:
                analyzer._log_host_memory()
        self.assertIn("analysis_host_memory_tight", " ".join(logs.output))

    def test_diagnostics_never_stop_an_analysis(self):
        """A missing psutil must cost a log line, not the run."""
        broken = mock.Mock()
        broken.virtual_memory.side_effect = RuntimeError("no /proc")
        with mock.patch.dict("sys.modules", {"psutil": broken}):
            with self.assertLogs("warrioriq.analysis", level="INFO"):
                analyzer._log_host_memory()          # must not raise


class GpuLoggingTests(unittest.TestCase):
    def test_no_cuda_is_reported_and_not_raised(self):
        with mock.patch("torch.cuda.is_available", return_value=False):
            with self.assertLogs("warrioriq.analysis", level="WARNING") as logs:
                analyzer._log_gpu_state()
        self.assertIn("analysis_gpu_absent", " ".join(logs.output))

    def test_a_contended_card_is_warned_about(self):
        with mock.patch("torch.cuda.is_available", return_value=True), \
             mock.patch("torch.cuda.current_device", return_value=0), \
             mock.patch("torch.cuda.get_device_name", return_value="Test GPU"), \
             mock.patch("torch.cuda.mem_get_info",
                        return_value=(1 * 1024 ** 3, 8 * 1024 ** 3)):
            with self.assertLogs("warrioriq.analysis", level="WARNING") as logs:
                analyzer._log_gpu_state()
        self.assertIn("analysis_gpu_contended", " ".join(logs.output))

    def test_a_free_card_logs_state_without_warning(self):
        with mock.patch("torch.cuda.is_available", return_value=True), \
             mock.patch("torch.cuda.current_device", return_value=0), \
             mock.patch("torch.cuda.get_device_name", return_value="Test GPU"), \
             mock.patch("torch.cuda.mem_get_info",
                        return_value=(7 * 1024 ** 3, 8 * 1024 ** 3)):
            with self.assertLogs("warrioriq.analysis", level="INFO") as logs:
                analyzer._log_gpu_state()
        joined = " ".join(logs.output)
        self.assertIn("analysis_gpu_state", joined)
        self.assertNotIn("analysis_gpu_contended", joined)


if __name__ == "__main__":
    unittest.main()
