"""Exclusive timing remains safe when NVML uses a different PID namespace."""
import unittest
from unittest.mock import Mock, patch

from experiments.rebuttal import control_benchmark as benchmark


class GPUOwnershipTests(unittest.TestCase):
    def test_host_pid_is_bound_after_creating_context(self):
        initialize = Mock()
        with patch.object(benchmark, 'gpu_process_ids', side_effect=[[], [195592]]):
            self.assertEqual(benchmark.bind_exclusive_cuda_process('GPU-test', initialize), 195592)
        initialize.assert_called_once_with()

    def test_occupied_gpu_is_rejected_before_creating_context(self):
        initialize = Mock()
        with patch.object(benchmark, 'gpu_process_ids', return_value=[888]):
            with self.assertRaises(AssertionError):
                benchmark.bind_exclusive_cuda_process('GPU-test', initialize)
        initialize.assert_not_called()

    def test_concurrent_process_and_missing_context_are_rejected(self):
        for owners in [[], [195592, 999]]:
            with self.subTest(owners=owners):
                with patch.object(benchmark, 'gpu_process_ids', side_effect=[[], owners]):
                    with self.assertRaises(AssertionError):
                        benchmark.bind_exclusive_cuda_process('GPU-test', Mock())

    def test_subsequent_occupancy_changes_fail(self):
        with patch.object(benchmark, 'gpu_process_ids', return_value=[195592]):
            benchmark.require_exclusive_gpu('GPU-test', 195592)
        for owners in [[], [999], [195592, 999]]:
            with self.subTest(owners=owners), patch.object(benchmark, 'gpu_process_ids', return_value=owners):
                with self.assertRaises(AssertionError):
                    benchmark.require_exclusive_gpu('GPU-test', 195592)

    def test_process_query_filters_the_requested_physical_gpu(self):
        with patch.object(benchmark.subprocess, 'check_output', return_value='GPU-other, 999\nGPU-test, 195592\n'):
            self.assertEqual(benchmark.gpu_process_ids('GPU-test'), [195592])


if __name__ == '__main__':
    unittest.main()
