import unittest

from scripts.benchmark_gpa import raise_worker_failures


class BenchmarkGpaFailureTest(unittest.TestCase):
    def test_parallel_worker_failures_raise_after_collection(self):
        failures = [
            (
                {"model": "scgpt", "method_name": "deepruot", "params": {"dim": 10}},
                RuntimeError("GPU failure"),
            ),
            (
                {"model": "hvg", "method_name": "deepruot", "params": {"dim": 10}},
                ValueError("bad output"),
            ),
        ]

        with self.assertRaisesRegex(
            RuntimeError,
            r"2 benchmark worker\(s\) failed.*scgpt.*GPU failure.*hvg.*bad output",
        ):
            raise_worker_failures(failures)

    def test_no_worker_failures_preserves_success(self):
        raise_worker_failures([])


if __name__ == "__main__":
    unittest.main()
