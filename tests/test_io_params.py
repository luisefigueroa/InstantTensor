import os
import unittest
import warnings
from types import SimpleNamespace
from contextlib import ExitStack
from unittest import mock

import torch

import instanttensor._cpu_count as cpu_count_impl
import instanttensor._impl as impl


IO_ENV_VARS = [
    "INSTANTTENSOR_BACKEND",
    "INSTANTTENSOR_CHUNK_SIZE",
    "INSTANTTENSOR_CONCURRENCY",
    "INSTANTTENSOR_IO_DEPTH",
    "INSTANTTENSOR_MAX_FREE_MEM_USAGE",
    "INSTANTTENSOR_BUFFER_SIZE",
    "INSTANTTENSOR_MEMORY_BUDGET_SOURCE",
]


class CPUCountTest(unittest.TestCase):
    def test_cpu_count_uses_smallest_constraint(self):
        with mock.patch.object(cpu_count_impl.os, "cpu_count", return_value=64), \
             mock.patch.object(
                 cpu_count_impl.os, "sched_getaffinity", return_value=set(range(8))
             ), \
             mock.patch.object(cpu_count_impl, "_cgroup_cpu_limit", return_value=3):
            self.assertEqual(cpu_count_impl.cpu_count(), 3)

    def test_cpu_count_is_at_least_one(self):
        with mock.patch.object(cpu_count_impl.os, "cpu_count", return_value=None), \
             mock.patch.object(cpu_count_impl.os, "sched_getaffinity", return_value=set()), \
             mock.patch.object(cpu_count_impl, "_cgroup_cpu_limit", return_value=None):
            self.assertEqual(cpu_count_impl.cpu_count(), 1)

    def test_cpu_count_cgroup_v2(self):
        cpu_max = mock.mock_open(read_data="150000 100000")
        with mock.patch.object(cpu_count_impl.os.path, "exists", return_value=True), \
             mock.patch("builtins.open", cpu_max):
            self.assertEqual(cpu_count_impl._cgroup_cpu_limit(), 2)

    def test_cpu_count_cgroup_v1(self):
        quota_file = mock.mock_open(read_data="150000").return_value
        period_file = mock.mock_open(read_data="100000").return_value
        exists = lambda path: path != cpu_count_impl._CGROUP_V2_CPU_MAX
        with mock.patch.object(cpu_count_impl.os.path, "exists", side_effect=exists), \
             mock.patch("builtins.open", side_effect=[quota_file, period_file]):
            self.assertEqual(cpu_count_impl._cgroup_cpu_limit(), 2)

    def test_cpu_count_cgroup_unlimited(self):
        cpu_max = mock.mock_open(read_data="max 100000")
        with mock.patch.object(cpu_count_impl.os.path, "exists", return_value=True), \
             mock.patch("builtins.open", cpu_max):
            self.assertIsNone(cpu_count_impl._cgroup_cpu_limit())


class IOParamsTest(unittest.TestCase):
    def setUp(self):
        self.saved_env = {name: os.environ.pop(name, None) for name in IO_ENV_VARS}

    def tearDown(self):
        for name, value in self.saved_env.items():
            os.environ.pop(name, None)
            if value is not None:
                os.environ[name] = value

    def determine_io_params(
        self,
        *,
        selected_backend,
        in_memory,
        world_size=1,
        chunk_size=None,
        concurrency=None,
        io_depth=None,
        buffer_size=None,
        free_bytes=1 << 50,
        max_free_mem_usage=1.0,
        budget_source=None,
        properties=None,
        meminfo=None,
        platform="linux",
        peer_budget=None,
    ):
        loader = impl.safe_open.__new__(impl.safe_open)
        loader.filename = ["model.safetensors"]
        loader.world_size = world_size
        loader.process_group = None
        loader.device = torch.device("cuda:0")
        loader.device_idx = 0
        loader.loader_handle = None
        if peer_budget is not None:
            loader.process_group = object()

        if properties is None:
            properties = SimpleNamespace(
                name="NVIDIA GB10", major=12, minor=1, is_integrated=True,
            )
        if meminfo is None:
            meminfo = "MemTotal: 131072000 kB\nMemAvailable: 28311552 kB\nSwapFree: 999999999 kB\n"
        if budget_source is not None:
            os.environ["INSTANTTENSOR_MEMORY_BUDGET_SOURCE"] = budget_source

        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(impl, "file_in_memory", return_value=in_memory))
            stack.enter_context(mock.patch.object(impl, "select_backend", return_value=selected_backend))
            stack.enter_context(mock.patch.object(impl, "cpu_count", return_value=64))
            stack.enter_context(mock.patch.object(impl.sys, "platform", platform))
            self.properties_query = stack.enter_context(mock.patch.object(
                impl.torch.cuda, "get_device_properties", return_value=properties,
            ))
            self.meminfo_read = stack.enter_context(mock.patch(
                "builtins.open", mock.mock_open(read_data=meminfo),
            ))
            if peer_budget is not None:
                original_tensor = torch.tensor
                stack.enter_context(mock.patch.object(
                    impl.torch, "tensor",
                    side_effect=lambda values, **kwargs: original_tensor(values),
                ))
                def reduce_min(tensor, *, op, group):
                    self.assertEqual(op, torch.distributed.ReduceOp.MIN)
                    self.assertIs(group, loader.process_group)
                    self.local_collective_budget = tensor.item()
                    tensor.fill_(min(tensor.item(), peer_budget))
                stack.enter_context(mock.patch.object(
                    impl.dist, "all_reduce", side_effect=reduce_min,
                ))
            stack.enter_context(mock.patch.object(
                impl.torch.cuda,
                "mem_get_info",
                return_value=(free_bytes, free_bytes),
            ))
            config = impl._resolve_open_config(
                buffer_size=buffer_size,
                chunk_size=chunk_size,
                concurrency=concurrency,
                io_depth=io_depth,
                max_free_mem_usage=max_free_mem_usage,
                backend=selected_backend,
            )
            loader._determine_io_params(config)
        return loader

    def test_default_budget_never_reads_host_metadata(self):
        for buffer_size in (None, 1342177280):
            with self.subTest(buffer_size=buffer_size):
                loader = self.determine_io_params(
                    selected_backend=impl.Backend.AIO, in_memory=False,
                    buffer_size=buffer_size, chunk_size=1 << 20, io_depth=4,
                    free_bytes=1031131130, max_free_mem_usage=0.1,
                )
                self.assertEqual(loader._device_memory_budget, 103113113)
                self.properties_query.assert_not_called()
                self.meminfo_read.assert_not_called()
                if buffer_size is not None:
                    loader.tensor_sizes = [1249902592]
                    loader.total_tensor_size = 1249902592
                    with self.assertRaisesRegex(RuntimeError, "exceeds device memory budget"):
                        loader._finalize_buffer_size(buffer_size)

    def test_gb10_explicit_ring_uses_host_fraction_without_swap_credit(self):
        for integrated in (True, 1):
            with self.subTest(integrated=integrated):
                loader = self.determine_io_params(
                    selected_backend=impl.Backend.AIO, in_memory=False,
                    world_size=2, buffer_size=1342177280,
                    chunk_size=1 << 20, io_depth=4, free_bytes=1031131130,
                    max_free_mem_usage=0.1, budget_source="mem_available",
                    properties=SimpleNamespace(
                        name="NVIDIA GB10", major=12, minor=1,
                        is_integrated=integrated,
                    ),
                )
                # The fixture has exactly 27 GiB available. Integer division
                # supplies an independent expected budget for fraction 1/10.
                self.assertEqual(loader._device_memory_budget, (27 << 30) // 10)
                loader.tensor_sizes = [1249902592, 128 << 20]
                loader.total_tensor_size = sum(loader.tensor_sizes)
                loader._finalize_buffer_size(1342177280)
                self.assertEqual(loader.buffer_size, 1342177280)
                self.assertEqual(loader.io_depth, 4)

    def test_host_budget_rejects_insufficient_ring_headroom_before_open(self):
        with mock.patch.object(impl._C, "open") as native_open:
            loader = self.determine_io_params(
                selected_backend=impl.Backend.AIO, in_memory=False,
                buffer_size=1342177280, chunk_size=1 << 20, io_depth=4,
                max_free_mem_usage=0.1, budget_source="mem_available",
                meminfo="MemTotal: 131072000 kB\nMemAvailable: 5242880 kB\n",
            )
            loader.tensor_sizes = [1249902592]
            loader.total_tensor_size = 1249902592
            with self.assertRaisesRegex(RuntimeError, "exceeds device memory budget"):
                loader._finalize_buffer_size(1342177280)
            native_open.assert_not_called()

    def test_bad_host_metadata_contributes_zero_to_peer_minimum(self):
        samples = [
            "MemTotal: 10000 kB\n",
            "MemAvailable: 1000 kB\n",
            "MemTotal: 0 kB\nMemAvailable: 0 kB\n",
            "MemTotal: 10000 kB\nMemAvailable: -1 kB\n",
            "MemTotal: 10000 kB\nMemAvailable: 10001 kB\n",
            "MemTotal: 10000 kB\nMemAvailable: 1000 MB\n",
            "MemTotal: 10000 kB\nMemAvailable: x kB\n",
            "MemTotal: 10000 kB\nMemAvailable: 1000 kB\nMemAvailable: 1000 kB\n",
            "MemTotal: 999999999999999999999 kB\nMemAvailable: 1 kB\n",
        ]
        for sample in samples:
            with self.subTest(sample=sample):
                with self.assertRaisesRegex(RuntimeError, "Cannot select memory budget"):
                    self.determine_io_params(
                        selected_backend=impl.Backend.AIO, in_memory=False,
                        world_size=2, buffer_size=1342177280,
                        chunk_size=1 << 20, io_depth=4,
                        budget_source="mem_available", meminfo=sample,
                        peer_budget=2 << 30,
                    )
                self.assertEqual(self.local_collective_budget, 0)

    def test_unreadable_host_metadata_contributes_zero(self):
        with mock.patch.object(impl, "_gb10_host_available_bytes", side_effect=OSError("unreadable meminfo")):
            with self.assertRaisesRegex(RuntimeError, "unreadable meminfo"):
                self.determine_io_params(
                    selected_backend=impl.Backend.AIO, in_memory=False,
                    world_size=2, buffer_size=1342177280,
                    chunk_size=1 << 20, io_depth=4,
                    budget_source="mem_available", peer_budget=2 << 30,
                )
        self.assertEqual(self.local_collective_budget, 0)

    def test_host_budget_requires_exact_hardware_metadata(self):
        properties = [
            SimpleNamespace(name="NVIDIA GB10", major=12, minor=0, is_integrated=True),
            SimpleNamespace(name="NVIDIA RTX PRO 6000", major=12, minor=1, is_integrated=True),
            SimpleNamespace(name="NVIDIA GB10", major=12, minor=1, is_integrated=False),
            SimpleNamespace(name="NVIDIA GB10", major=12, minor=1, is_integrated="1"),
            SimpleNamespace(name="NVIDIA GB10", major=12, minor=1),
        ]
        for value in properties:
            with self.subTest(properties=value):
                with self.assertRaisesRegex(RuntimeError, "Cannot select memory budget"):
                    self.determine_io_params(
                        selected_backend=impl.Backend.AIO, in_memory=False,
                        buffer_size=1342177280, chunk_size=1 << 20, io_depth=4,
                        budget_source="mem_available", properties=value,
                    )
                self.meminfo_read.assert_not_called()

    def test_host_budget_requires_linux_and_explicit_ring(self):
        for platform, buffer_size in (("win32", 1342177280), ("linux", None)):
            with self.subTest(platform=platform, buffer_size=buffer_size):
                with self.assertRaisesRegex(RuntimeError, "mem_available requires"):
                    self.determine_io_params(
                        selected_backend=impl.Backend.AIO, in_memory=False,
                        buffer_size=buffer_size, chunk_size=1 << 20, io_depth=4,
                        budget_source="mem_available", platform=platform,
                    )
                self.meminfo_read.assert_not_called()

    def test_memory_fraction_rejects_invalid_values(self):
        for value in (0, -0.1, 1.01, float("nan"), float("inf"), -float("inf")):
            with self.subTest(fraction=value):
                with self.assertRaisesRegex(ValueError, "finite.*0 < value <= 1"):
                    self.determine_io_params(
                        selected_backend=impl.Backend.AIO, in_memory=False,
                        buffer_size=1342177280, chunk_size=1 << 20, io_depth=4,
                        max_free_mem_usage=value,
                    )

    def test_unknown_budget_source_rejects_after_zero_peer_minimum(self):
        with self.assertRaisesRegex(RuntimeError, "must be cuda_free or mem_available"):
            self.determine_io_params(
                selected_backend=impl.Backend.AIO, in_memory=False,
                world_size=2, buffer_size=1342177280,
                chunk_size=1 << 20, io_depth=4,
                budget_source="host_available", peer_budget=2 << 30,
            )
        self.assertEqual(self.local_collective_budget, 0)

    def test_host_fraction_environment_retains_its_meaning(self):
        os.environ["INSTANTTENSOR_MAX_FREE_MEM_USAGE"] = "0.1"
        loader = self.determine_io_params(
            selected_backend=impl.Backend.AIO, in_memory=False,
            buffer_size=1342177280, chunk_size=1 << 20, io_depth=4,
            max_free_mem_usage=None, budget_source="mem_available",
        )
        self.assertEqual(loader._device_memory_budget, (27 << 30) // 10)

    def test_host_budget_uses_smaller_peer_and_rejects_invalid_peer(self):
        for peer_budget in (2 << 30, 64 << 20, 0):
            with self.subTest(peer_budget=peer_budget):
                options = dict(
                    selected_backend=impl.Backend.AIO, in_memory=False,
                    world_size=2, buffer_size=1342177280,
                    chunk_size=1 << 20, io_depth=4,
                    max_free_mem_usage=0.1, budget_source="mem_available",
                    peer_budget=peer_budget,
                )
                if peer_budget == 0:
                    with self.assertRaisesRegex(RuntimeError, "too small for one I/O operation"):
                        self.determine_io_params(**options)
                    self.assertEqual(self.local_collective_budget, (27 << 30) // 10)
                    continue
                loader = self.determine_io_params(**options)
                self.assertEqual(loader._device_memory_budget, peer_budget)
                loader.tensor_sizes = [1249902592]
                loader.total_tensor_size = 1249902592
                if peer_budget < 1342177280:
                    with self.assertRaisesRegex(RuntimeError, "exceeds device memory budget"):
                        loader._finalize_buffer_size(1342177280)
                else:
                    loader._finalize_buffer_size(1342177280)

    def test_largest_tensor_enlargement_still_obeys_host_budget(self):
        loader = self.determine_io_params(
            selected_backend=impl.Backend.AIO, in_memory=False,
            buffer_size=1 << 20, chunk_size=1 << 20, io_depth=1,
            max_free_mem_usage=0.1, budget_source="mem_available",
        )
        loader.tensor_sizes = [1249902592]
        loader.total_tensor_size = 1249902592
        with self.assertWarnsRegex(RuntimeWarning, "match the largest tensor"):
            loader._finalize_buffer_size(1 << 20)
        self.assertEqual(loader.buffer_size, 1249902592)
        loader._device_memory_budget = 1 << 30
        with self.assertWarnsRegex(RuntimeWarning, "match the largest tensor"):
            with self.assertRaisesRegex(RuntimeError, "exceeds device memory budget"):
                loader._finalize_buffer_size(1 << 20)

    def test_native_allocator_failure_propagates_after_host_admission(self):
        loader = self.determine_io_params(
            selected_backend=impl.Backend.AIO, in_memory=False,
            buffer_size=1342177280, chunk_size=1 << 20, io_depth=4,
            max_free_mem_usage=0.1, budget_source="mem_available",
        )
        loader.tensor_sizes = [1249902592]
        loader.total_tensor_size = 1249902592
        loader.tensor_offsets = []
        loader._finalize_buffer_size(1342177280)
        with mock.patch.object(impl._C, "open", side_effect=RuntimeError("native allocation failed")) as native_open:
            with self.assertRaisesRegex(RuntimeError, "native allocation failed"):
                loader._open()
            native_open.assert_called_once()
        self.assertIsNone(loader.loader_handle)

    def test_native_configuration_is_the_single_source_of_truth(self):
        self.assertEqual(
            {backend.name: backend.value for backend in impl.Backend},
            impl._C.backend_values(),
        )
        self.assertEqual(
            impl.MAX_IO_DEPTH,
            impl._C.MAX_IO_DEPTH,
        )

        page_size = os.sysconf("SC_PAGE_SIZE")
        self.assertEqual(
            impl.required_buffer_size_for_io(page_size + 1, 3, 2),
            2 * page_size * 3 * 2,
        )

    def test_mmap_default_depth_includes_worker_concurrency(self):
        loader = self.determine_io_params(
            selected_backend=impl.Backend.MMAP,
            in_memory=True,
        )

        self.assertEqual(loader.chunk_size, 2 * 1024 * 1024)
        self.assertEqual(loader.concurrency, 32)
        self.assertEqual(loader.io_depth, 3 * loader.concurrency)

    def test_cufile_default_depth_includes_worker_concurrency(self):
        loader = self.determine_io_params(
            selected_backend=impl.Backend.CUFILE,
            in_memory=False,
            world_size=2,
        )

        self.assertEqual(loader.chunk_size, 8 * 1024 * 1024)
        self.assertEqual(loader.concurrency, 16)
        self.assertEqual(loader.io_depth, 2 * loader.concurrency)

    def test_native_async_depth_does_not_depend_on_concurrency(self):
        with self.assertWarnsRegex(RuntimeWarning, "does not support concurrency"):
            loader = self.determine_io_params(
                selected_backend=impl.Backend.URING,
                in_memory=False,
                world_size=2,
                concurrency=37,
            )

        self.assertEqual(loader.concurrency, 0)
        self.assertEqual(loader.io_depth, 256)

    def test_native_async_default_concurrency_is_unused(self):
        loader = self.determine_io_params(
            selected_backend=impl.Backend.URING,
            in_memory=False,
        )

        self.assertEqual(loader.concurrency, 0)
        self.assertEqual(loader.io_depth, 512)

    def test_buffered_native_async_default_concurrency_is_unused(self):
        loader = self.determine_io_params(
            selected_backend=impl.Backend.URING_BUFFERED,
            in_memory=True,
        )

        self.assertEqual(loader.concurrency, 0)
        self.assertEqual(loader.io_depth, 32)
        loader.tensor_sizes = [1]
        loader._finalize_buffer_size(None)
        self.assertEqual(
            loader.buffer_size,
            loader.chunk_size * loader.io_depth * loader.world_size,
        )

    def test_in_memory_native_async_depth_does_not_depend_on_concurrency(self):
        with self.assertWarnsRegex(RuntimeWarning, "does not support concurrency"):
            loader = self.determine_io_params(
                selected_backend=impl.Backend.URING_BUFFERED,
                in_memory=True,
                concurrency=37,
            )

        self.assertEqual(loader.concurrency, 0)
        self.assertEqual(loader.io_depth, 32)

    def test_memory_limit_shrinks_depth_not_worker_concurrency(self):
        chunk_size = 8 * 1024 * 1024
        with self.assertWarnsRegex(RuntimeWarning, "Shrink io_depth"):
            loader = self.determine_io_params(
                selected_backend=impl.Backend.MMAP,
                in_memory=True,
                chunk_size=chunk_size,
                concurrency=4,
                io_depth=10,
                free_bytes=chunk_size * 5,
            )

        self.assertEqual(loader.concurrency, 4)
        self.assertEqual(loader.io_depth, 5)

    def test_worker_backends_reject_zero_concurrency_before_io_depth(self):
        for backend in (impl.Backend.MMAP, impl.Backend.CUFILE):
            for io_depth in (None, 0, 1):
                with self.subTest(backend=backend.name, io_depth=io_depth):
                    with self.assertRaisesRegex(ValueError, "concurrency must be greater than zero"):
                        self.determine_io_params(
                            selected_backend=backend, in_memory=False,
                            concurrency=0, io_depth=io_depth,
                        )

    def test_all_backends_reject_negative_concurrency_before_io_depth(self):
        for backend in impl.Backend:
            with self.subTest(backend=backend.name):
                with self.assertRaisesRegex(ValueError, "concurrency must not be negative"):
                    self.determine_io_params(
                        selected_backend=backend, in_memory=False,
                        concurrency=-1, io_depth=0,
                    )

    def test_worker_backends_preserve_positive_concurrency(self):
        for backend, depth_factor in ((impl.Backend.MMAP, 3), (impl.Backend.CUFILE, 2)):
            with self.subTest(backend=backend.name):
                with warnings.catch_warnings(record=True) as emitted:
                    warnings.simplefilter("always")
                    loader = self.determine_io_params(
                        selected_backend=backend, in_memory=False, concurrency=5,
                    )
                self.assertEqual(loader.concurrency, 5)
                self.assertEqual(loader.io_depth, depth_factor * 5)
                self.assertEqual(emitted, [])

    def test_async_backends_warn_and_override_positive_concurrency(self):
        for backend in (impl.Backend.AIO, impl.Backend.URING,
                        impl.Backend.AIO_BUFFERED, impl.Backend.URING_BUFFERED):
            with self.subTest(backend=backend.name):
                with self.assertWarnsRegex(RuntimeWarning, "concurrency=5 to 0"):
                    loader = self.determine_io_params(
                        selected_backend=backend, in_memory=False,
                        concurrency=5, io_depth=7,
                    )
                self.assertEqual(loader.concurrency, 0)
                self.assertEqual(loader.io_depth, 7)

    def test_async_backends_accept_zero_and_default_without_warning(self):
        for backend in (impl.Backend.AIO, impl.Backend.URING,
                        impl.Backend.AIO_BUFFERED, impl.Backend.URING_BUFFERED):
            for concurrency in (None, 0):
                with self.subTest(backend=backend.name, concurrency=concurrency):
                    with warnings.catch_warnings(record=True) as emitted:
                        warnings.simplefilter("always")
                        loader = self.determine_io_params(
                            selected_backend=backend, in_memory=False,
                            concurrency=concurrency,
                        )
                    self.assertEqual(loader.concurrency, 0)
                    self.assertEqual(emitted, [])

    def test_concurrency_environment_uses_same_validation(self):
        os.environ["INSTANTTENSOR_CONCURRENCY"] = "0"
        for backend in (impl.Backend.MMAP, impl.Backend.CUFILE):
            with self.subTest(backend=backend.name):
                with self.assertRaisesRegex(ValueError, "concurrency must be greater than zero"):
                    self.determine_io_params(selected_backend=backend, in_memory=False)
        os.environ["INSTANTTENSOR_CONCURRENCY"] = "5"
        with self.assertWarnsRegex(RuntimeWarning, "concurrency=5 to 0"):
            loader = self.determine_io_params(selected_backend=impl.Backend.AIO, in_memory=False)
        self.assertEqual(loader.concurrency, 0)

    def test_io_depth_cannot_exceed_executor_capacity(self):
        with self.assertRaisesRegex(ValueError, "io_depth must not exceed"):
            self.determine_io_params(
                selected_backend=impl.Backend.URING,
                in_memory=False,
                io_depth=impl.MAX_IO_DEPTH + 1,
            )

    def test_explicit_buffer_and_depth_must_be_compatible(self):
        chunk_size = 8 * 1024 * 1024
        with self.assertRaisesRegex(ValueError, "too small for io_depth=4"):
            self.determine_io_params(
                selected_backend=impl.Backend.URING,
                in_memory=False,
                chunk_size=chunk_size,
                io_depth=4,
                buffer_size=3 * chunk_size,
            )

    def test_explicit_buffer_shrinks_default_depth(self):
        chunk_size = 8 * 1024 * 1024
        with self.assertWarnsRegex(RuntimeWarning, "to fit buffer_size"):
            loader = self.determine_io_params(
                selected_backend=impl.Backend.URING,
                in_memory=False,
                chunk_size=chunk_size,
                buffer_size=4 * chunk_size,
            )

        self.assertEqual(loader.io_depth, 4)

    def test_explicit_buffer_must_fit_one_io_operation(self):
        chunk_size = 8 * 1024 * 1024
        with self.assertRaisesRegex(ValueError, "too small for one I/O operation"):
            self.determine_io_params(
                selected_backend=impl.Backend.URING,
                in_memory=False,
                chunk_size=chunk_size,
                buffer_size=chunk_size - 1,
            )

    def test_buffer_limit_uses_page_aligned_chunk_size(self):
        page_size = os.sysconf("SC_PAGE_SIZE")
        chunk_size = page_size + 1
        with self.assertRaisesRegex(ValueError, "too small for io_depth=2"):
            self.determine_io_params(
                selected_backend=impl.Backend.URING,
                in_memory=False,
                chunk_size=chunk_size,
                io_depth=2,
                buffer_size=3 * page_size,
            )

    def test_memory_limit_uses_page_aligned_chunk_size(self):
        page_size = os.sysconf("SC_PAGE_SIZE")
        chunk_size = page_size + 1
        with self.assertWarnsRegex(RuntimeWarning, "due to memory limit"):
            loader = self.determine_io_params(
                selected_backend=impl.Backend.URING,
                in_memory=False,
                chunk_size=chunk_size,
                io_depth=2,
                free_bytes=3 * page_size,
            )

        self.assertEqual(loader.io_depth, 1)

    def test_memory_limit_must_fit_one_io_operation(self):
        chunk_size = 8 * 1024 * 1024
        with self.assertRaisesRegex(
            RuntimeError, "too small for one I/O operation",
        ):
            self.determine_io_params(
                selected_backend=impl.Backend.URING,
                in_memory=False,
                chunk_size=chunk_size,
                free_bytes=chunk_size - 1,
            )

    def test_environment_buffer_and_depth_are_explicit(self):
        chunk_size = 8 * 1024 * 1024
        os.environ["INSTANTTENSOR_BUFFER_SIZE"] = str(3 * chunk_size)
        os.environ["INSTANTTENSOR_IO_DEPTH"] = "4"

        with self.assertRaisesRegex(ValueError, "too small for io_depth=4"):
            self.determine_io_params(
                selected_backend=impl.Backend.URING,
                in_memory=False,
                chunk_size=chunk_size,
            )

    def test_explicit_buffer_is_not_shrunk_below_io_requirement(self):
        chunk_size = 8 * 1024 * 1024
        loader = self.determine_io_params(
            selected_backend=impl.Backend.URING,
            in_memory=False,
            chunk_size=chunk_size,
            io_depth=3,
            buffer_size=4 * chunk_size,
        )
        loader.tensor_sizes = [1]
        loader.total_tensor_size = 1

        with self.assertWarnsRegex(RuntimeWarning, "Shrink buffer size"):
            loader._finalize_buffer_size(4 * chunk_size)

        self.assertEqual(loader.buffer_size, 3 * chunk_size)

    def test_final_buffer_must_fit_device_memory_budget(self):
        chunk_size = 8 * 1024 * 1024
        loader = self.determine_io_params(
            selected_backend=impl.Backend.URING,
            in_memory=False,
            chunk_size=chunk_size,
            io_depth=1,
            free_bytes=2 * chunk_size,
        )
        loader.tensor_sizes = [3 * chunk_size]
        loader.total_tensor_size = 3 * chunk_size

        with self.assertRaisesRegex(
            RuntimeError, "exceeds device memory budget",
        ):
            loader._finalize_buffer_size(None)

    def test_buffer_environment_is_resolved_once(self):
        chunk_size = 8 * 1024 * 1024
        configured_buffer_size = 4 * chunk_size
        loader = impl.safe_open.__new__(impl.safe_open)
        loader.filename = ["model.safetensors"]
        loader.world_size = 1
        loader.process_group = None
        loader.device = torch.device("cuda:0")
        loader.tensor_sizes = [1]
        loader.total_tensor_size = 1

        with ExitStack() as stack:
            env_buffer_size = stack.enter_context(mock.patch.object(
                impl, "env_buffer_size", return_value=configured_buffer_size,
            ))
            stack.enter_context(mock.patch.object(impl, "file_in_memory", return_value=False))
            stack.enter_context(mock.patch.object(
                impl, "select_backend", return_value=impl.Backend.URING,
            ))
            stack.enter_context(mock.patch.object(
                impl.torch.cuda,
                "mem_get_info",
                return_value=(1 << 50, 1 << 50),
            ))

            config = impl._resolve_open_config(
                buffer_size=None,
                chunk_size=chunk_size,
                concurrency=0,
                io_depth=4,
                max_free_mem_usage=1.0,
                backend=impl.Backend.URING,
            )
            loader._determine_io_params(config)
            loader._finalize_buffer_size(config.buffer_size)

        env_buffer_size.assert_called_once_with()

    def test_uring_requires_linux_5_6(self):
        backend_status = mock.Mock(return_value=(
            False,
            "io_uring requires Linux kernel 5.6 or newer; the detected kernel "
            "version is 5.5.0.",
            "",
        ))
        with mock.patch.object(
            impl._C, "backend_status", backend_status,
        ):
            with self.assertRaises(RuntimeError) as raised:
                impl.select_backend([impl.Backend.URING])

        self.assertEqual(
            str(raised.exception),
            "No available backend was found among candidates [URING]. "
            "io_uring requires Linux kernel 5.6 or newer; the detected kernel "
            "version is 5.5.0.",
        )
        backend_status.assert_called_once_with(impl.Backend.URING.value)

    def test_uring_warns_below_recommended_kernel(self):
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(
                impl._C,
                "backend_status",
                return_value=(
                    True,
                    "",
                    "io_uring on Linux 5.14 may be unstable; "
                    "Linux 5.15 or newer is recommended.",
                ),
            ))
            stack.enter_context(mock.patch.object(
                impl, "_emitted_backend_warnings", set(),
            ))

            with self.assertWarnsRegex(RuntimeWarning, "Linux 5.15 or newer"):
                backend = impl.select_backend([impl.Backend.URING])

        self.assertEqual(backend, impl.Backend.URING)

    def test_backend_warning_is_emitted_once(self):
        warning = (
            "io_uring on Linux 5.14 may be unstable; "
            "Linux 5.15 or newer is recommended."
        )
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(
                impl._C,
                "backend_status",
                return_value=(True, "", warning),
            ))
            stack.enter_context(mock.patch.object(
                impl, "_emitted_backend_warnings", set(),
            ))
            warn = stack.enter_context(mock.patch.object(impl.warnings, "warn"))

            impl.select_backend([impl.Backend.URING])
            impl.select_backend([impl.Backend.URING_BUFFERED])

        warn.assert_called_once()

    def test_uring_does_not_warn_on_recommended_kernel(self):
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(
                impl._C,
                "backend_status",
                return_value=(True, "", ""),
            ))
            warn = stack.enter_context(mock.patch.object(impl.warnings, "warn"))

            backend = impl.select_backend([impl.Backend.URING_BUFFERED])

        self.assertEqual(backend, impl.Backend.URING_BUFFERED)
        warn.assert_not_called()


if __name__ == "__main__":
    unittest.main()
