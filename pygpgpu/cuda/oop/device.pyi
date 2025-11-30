from uuid import UUID

from .cuobject import CUObject
from .context import Context
from ..driver import CUctx_flags


class Device(CUObject):

    def __init__(self, device_id:int)->None: ...

    def create_context(self, flags:CUctx_flags=CUctx_flags.CU_CTX_SCHED_AUTO)->Context: ...

    @property
    def unique_key(self)->str: ...

    @property
    def default_context(self)->Context: ...

    @property
    def name(self)->str: ...

    @property
    def total_memory(self)->int: ...

    @property
    def uuid(self)->UUID: ...

    @property
    def max_threads_per_block(self)->int:
        """Maximum number of threads per block"""
        ...

    @property
    def max_block_dim_x(self)->int:
        """Maximum block dimension X"""
        ...

    @property
    def max_block_dim_y(self)->int:
        """Maximum block dimension Y"""
        ...

    @property
    def max_block_dim_z(self)->int:
        """Maximum block dimension Z"""
        ...

    @property
    def max_grid_dim_x(self)->int:
        """Maximum grid dimension X"""
        ...

    @property
    def max_grid_dim_y(self)->int:
        """Maximum grid dimension Y"""
        ...

    @property
    def max_grid_dim_z(self)->int:
        """Maximum grid dimension Z"""
        ...

    @property
    def max_shared_memory_per_block(self)->int:
        """Maximum shared memory available per block in bytes"""
        ...

    @property
    def shared_memory_per_block(self)->int:
        """Deprecated, use def MAX_SHARED_MEMORY_PER_BLOCK"""
        ...

    @property
    def total_constant_memory(self)->int:
        """Memory available on device for __constant__ variables in a CUDA C kernel in bytes"""
        ...

    @property
    def warp_size(self)->int:
        """Warp size in threads"""
        ...

    @property
    def max_pitch(self)->int:
        """Maximum pitch in bytes allowed by memory copies"""
        ...

    @property
    def max_registers_per_block(self)->int:
        """Maximum number of 32-bit registers available per block"""
        ...

    @property
    def registers_per_block(self)->int:
        """Deprecated, use def MAX_REGISTERS_PER_BLOCK"""
        ...

    @property
    def clock_rate(self)->int:
        """Typical clock frequency in kilohertz"""
        ...

    @property
    def texture_alignment(self)->int:
        """Alignment requirement for textures"""
        ...

    @property
    def gpu_overlap(self)->int:
        """Device can possibly copy memory and execute a kernel concurrently. Deprecated. Use instead def ASYNC_ENGINE_COUNT."""
        ...

    @property
    def multiprocessor_count(self)->int:
        """Number of multiprocessors on device"""
        ...

    @property
    def kernel_exec_timeout(self)->int:
        """Specifies whether there is a run time limit on kernels"""
        ...

    @property
    def integrated(self)->int:
        """Device is integrated with host memory"""
        ...

    @property
    def can_map_host_memory(self)->int:
        """Device can map host memory into CUDA address space"""
        ...

    @property
    def compute_mode(self)->int:
        """Compute mode (See ::CUcomputemode for details)"""
        ...

    @property
    def maximum_texture1d_width(self)->int:
        """Maximum 1D texture width"""
        ...

    @property
    def maximum_texture2d_width(self)->int:
        """Maximum 2D texture width"""
        ...

    @property
    def maximum_texture2d_height(self)->int:
        """Maximum 2D texture height"""
        ...

    @property
    def maximum_texture3d_width(self)->int:
        """Maximum 3D texture width"""
        ...

    @property
    def maximum_texture3d_height(self)->int:
        """Maximum 3D texture height"""
        ...

    @property
    def maximum_texture3d_depth(self)->int:
        """Maximum 3D texture depth"""
        ...

    @property
    def maximum_texture2d_layered_width(self)->int:
        """Maximum 2D layered texture width"""
        ...

    @property
    def maximum_texture2d_layered_height(self)->int:
        """Maximum 2D layered texture height"""
        ...

    @property
    def maximum_texture2d_layered_layers(self)->int:
        """Maximum layers in a 2D layered texture"""
        ...

    @property
    def maximum_texture2d_array_width(self)->int:
        """Deprecated, use MAXIMUM_TEXTURE2D_LAYERED_WIDTH"""
        ...

    @property
    def maximum_texture2d_array_height(self)->int:
        """Deprecated, use MAXIMUM_TEXTURE2D_LAYERED_HEIGHT"""
        ...

    @property
    def maximum_texture2d_array_numslices(self)->int:
        """Deprecated, use MAXIMUM_TEXTURE2D_LAYERED_LAYERS"""
        ...

    @property
    def surface_alignment(self)->int:
        """Alignment requirement for surfaces"""
        ...

    @property
    def concurrent_kernels(self)->int:
        """Device can possibly execute multiple kernels concurrently"""
        ...

    @property
    def ecc_enabled(self)->int:
        """Device has ECC support enabled"""
        ...

    @property
    def pci_bus_id(self)->int:
        """PCI bus ID of the device"""
        ...

    @property
    def pci_device_id(self)->int:
        """PCI device ID of the device"""
        ...

    @property
    def tcc_driver(self)->int:
        """Device is using TCC driver model"""
        ...

    @property
    def memory_clock_rate(self)->int:
        """Peak memory clock frequency in kilohertz"""
        ...

    @property
    def global_memory_bus_width(self)->int:
        """Global memory bus width in bits"""
        ...

    @property
    def l2_cache_size(self)->int:
        """Size of L2 cache in bytes"""
        ...

    @property
    def max_threads_per_multiprocessor(self)->int:
        """Maximum resident threads per multiprocessor"""
        ...

    @property
    def async_engine_count(self)->int:
        """Number of asynchronous engines"""
        ...

    @property
    def unified_addressing(self)->int:
        """Device shares a unified address space with the host"""
        ...

    @property
    def maximum_texture1d_layered_width(self)->int:
        """Maximum 1D layered texture width"""
        ...

    @property
    def maximum_texture1d_layered_layers(self)->int:
        """Maximum layers in a 1D layered texture"""
        ...

    @property
    def can_tex2d_gather(self)->int:
        """Deprecated, do not use."""
        ...

    @property
    def maximum_texture2d_gather_width(self)->int:
        """Maximum 2D texture width if CUDA_ARRAY3D_TEXTURE_GATHER is set"""
        ...

    @property
    def maximum_texture2d_gather_height(self)->int:
        """Maximum 2D texture height if CUDA_ARRAY3D_TEXTURE_GATHER is set"""
        ...

    @property
    def maximum_texture3d_width_alternate(self)->int:
        """Alternate maximum 3D texture width"""
        ...

    @property
    def maximum_texture3d_height_alternate(self)->int:
        """Alternate maximum 3D texture height"""
        ...

    @property
    def maximum_texture3d_depth_alternate(self)->int:
        """Alternate maximum 3D texture depth"""
        ...

    @property
    def pci_domain_id(self)->int:
        """PCI domain ID of the device"""
        ...

    @property
    def texture_pitch_alignment(self)->int:
        """Pitch alignment requirement for textures"""
        ...

    @property
    def maximum_texturecubemap_width(self)->int:
        """Maximum cubemap texture width/height"""
        ...

    @property
    def maximum_texturecubemap_layered_width(self)->int:
        """Maximum cubemap layered texture width/height"""
        ...

    @property
    def maximum_texturecubemap_layered_layers(self)->int:
        """Maximum layers in a cubemap layered texture"""
        ...

    @property
    def maximum_surface1d_width(self)->int:
        """Maximum 1D surface width"""
        ...

    @property
    def maximum_surface2d_width(self)->int:
        """Maximum 2D surface width"""
        ...

    @property
    def maximum_surface2d_height(self)->int:
        """Maximum 2D surface height"""
        ...

    @property
    def maximum_surface3d_width(self)->int:
        """Maximum 3D surface width"""
        ...

    @property
    def maximum_surface3d_height(self)->int:
        """Maximum 3D surface height"""
        ...

    @property
    def maximum_surface3d_depth(self)->int:
        """Maximum 3D surface depth"""
        ...

    @property
    def maximum_surface1d_layered_width(self)->int:
        """Maximum 1D layered surface width"""
        ...

    @property
    def maximum_surface1d_layered_layers(self)->int:
        """Maximum layers in a 1D layered surface"""
        ...

    @property
    def maximum_surface2d_layered_width(self)->int:
        """Maximum 2D layered surface width"""
        ...

    @property
    def maximum_surface2d_layered_height(self)->int:
        """Maximum 2D layered surface height"""
        ...

    @property
    def maximum_surface2d_layered_layers(self)->int:
        """Maximum layers in a 2D layered surface"""
        ...

    @property
    def maximum_surfacecubemap_width(self)->int:
        """Maximum cubemap surface width"""
        ...

    @property
    def maximum_surfacecubemap_layered_width(self)->int:
        """Maximum cubemap layered surface width"""
        ...

    @property
    def maximum_surfacecubemap_layered_layers(self)->int:
        """Maximum layers in a cubemap layered surface"""
        ...

    @property
    def maximum_texture1d_linear_width(self)->int:
        """Deprecated, do not use. Use cudaDeviceGetTexture1DLinearMaxWidth() or cuDeviceGetTexture1DLinearMaxWidth() instead."""
        ...

    @property
    def maximum_texture2d_linear_width(self)->int:
        """Maximum 2D linear texture width"""
        ...

    @property
    def maximum_texture2d_linear_height(self)->int:
        """Maximum 2D linear texture height"""
        ...

    @property
    def maximum_texture2d_linear_pitch(self)->int:
        """Maximum 2D linear texture pitch in bytes"""
        ...

    @property
    def maximum_texture2d_mipmapped_width(self)->int:
        """Maximum mipmapped 2D texture width"""
        ...

    @property
    def maximum_texture2d_mipmapped_height(self)->int:
        """Maximum mipmapped 2D texture height"""
        ...

    @property
    def compute_capability_major(self)->int:
        """Major compute capability version number"""
        ...

    @property
    def compute_capability_minor(self)->int:
        """Minor compute capability version number"""
        ...

    @property
    def maximum_texture1d_mipmapped_width(self)->int:
        """Maximum mipmapped 1D texture width"""
        ...

    @property
    def stream_priorities_supported(self)->int:
        """Device supports stream priorities"""
        ...

    @property
    def global_l1_cache_supported(self)->int:
        """Device supports caching globals in L1"""
        ...

    @property
    def local_l1_cache_supported(self)->int:
        """Device supports caching locals in L1"""
        ...

    @property
    def max_shared_memory_per_multiprocessor(self)->int:
        """Maximum shared memory available per multiprocessor in bytes"""
        ...

    @property
    def max_registers_per_multiprocessor(self)->int:
        """Maximum number of 32-bit registers available per multiprocessor"""
        ...

    @property
    def managed_memory(self)->int:
        """Device can allocate managed memory on this system"""
        ...

    @property
    def multi_gpu_board(self)->int:
        """Device is on a multi-GPU board"""
        ...

    @property
    def multi_gpu_board_group_id(self)->int:
        """Unique id for a group of devices on the same multi-GPU board"""
        ...

    @property
    def host_native_atomic_supported(self)->int:
        """Link between the device and the host supports native atomic operations (this is a placeholder attribute, and is not supported on any current hardware)*/"""
        ...

    @property
    def single_to_double_precision_perf_ratio(self)->int:
        """Ratio of single precision performance (in floating-point operations per second) to double precision performance"""
        ...

    @property
    def pageable_memory_access(self)->int:
        """Device supports coherently accessing pageable memory without calling cudaHostRegister on it"""
        ...

    @property
    def concurrent_managed_access(self)->int:
        """Device can coherently access managed memory concurrently with the CPU"""
        ...

    @property
    def compute_preemption_supported(self)->int:
        """Device supports compute preemption."""
        ...

    @property
    def can_use_host_pointer_for_registered_mem(self)->int:
        """Device can access host registered memory at the same virtual address as the CPU"""
        ...

    @property
    def can_use_stream_mem_ops_v1(self)->int:
        """Deprecated, along with v1 MemOps API, ::cuStreamBatchMemOp and related APIs are supported."""
        ...

    @property
    def can_use_64_bit_stream_mem_ops_v1(self)->int:
        """Deprecated, along with v1 MemOps API, 64-bit operations are supported in ::cuStreamBatchMemOp and related APIs."""
        ...

    @property
    def can_use_stream_wait_value_nor_v1(self)->int:
        """Deprecated, along with v1 MemOps API, ::CU_STREAM_WAIT_VALUE_NOR is supported."""
        ...

    @property
    def cooperative_launch(self)->int:
        """Device supports launching cooperative kernels via ::cuLaunchCooperativeKernel"""
        ...

    @property
    def cooperative_multi_device_launch(self)->int:
        """Deprecated, ::cuLaunchCooperativeKernelMultiDevice is deprecated."""
        ...

    @property
    def max_shared_memory_per_block_optin(self)->int:
        """Maximum optin shared memory per block"""
        ...

    @property
    def can_flush_remote_writes(self)->int:
        """The ::CU_STREAM_WAIT_VALUE_FLUSH flag and the ::CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES MemOp are supported on the device. See \ref CUDA_MEMOP for additional details."""
        ...

    @property
    def host_register_supported(self)->int:
        """Device supports host memory registration via ::cudaHostRegister."""
        ...

    @property
    def pageable_memory_access_uses_host_page_tables(self)->int:
        """Device accesses pageable memory via the host's page tables."""
        ...

    @property
    def direct_managed_mem_access_from_host(self)->int:
        """The host can directly access managed memory on the device without migration."""
        ...

    @property
    def virtual_address_management_supported(self)->int:
        """Deprecated, Use def VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED*/"""
        ...

    @property
    def virtual_memory_management_supported(self)->int:
        """Device supports virtual memory management APIs like ::cuMemAddressReserve, ::cuMemCreate, ::cuMemMap and related APIs"""
        ...

    @property
    def handle_type_posix_file_descriptor_supported(self)->int:
        """Device supports exporting memory to a posix file descriptor with ::cuMemExportToShareableHandle, if requested via ::cuMemCreate"""
        ...

    @property
    def handle_type_win32_handle_supported(self)->int:
        """Device supports exporting memory to a Win32 NT handle with ::cuMemExportToShareableHandle, if requested via ::cuMemCreate"""
        ...

    @property
    def handle_type_win32_kmt_handle_supported(self)->int:
        """Device supports exporting memory to a Win32 KMT handle with ::cuMemExportToShareableHandle, if requested via ::cuMemCreate"""
        ...

    @property
    def max_blocks_per_multiprocessor(self)->int:
        """Maximum number of blocks per multiprocessor"""
        ...

    @property
    def generic_compression_supported(self)->int:
        """Device supports compression of memory"""
        ...

    @property
    def max_persisting_l2_cache_size(self)->int:
        """Maximum L2 persisting lines capacity setting in bytes."""
        ...

    @property
    def max_access_policy_window_size(self)->int:
        """Maximum value of CUaccessPolicyWindow::num_bytes."""
        ...

    @property
    def gpu_direct_rdma_with_cuda_vmm_supported(self)->int:
        """Device supports specifying the GPUDirect RDMA flag with ::cuMemCreate"""
        ...

    @property
    def reserved_shared_memory_per_block(self)->int:
        """Shared memory reserved by CUDA driver per block in bytes"""
        ...

    @property
    def sparse_cuda_array_supported(self)->int:
        """Device supports sparse CUDA arrays and sparse CUDA mipmapped arrays"""
        ...

    @property
    def read_only_host_register_supported(self)->int:
        """Device supports using the ::cuMemHostRegister flag ::CU_MEMHOSTERGISTER_READ_ONLY to register memory that must be mapped as read-only to the GPU"""
        ...

    @property
    def timeline_semaphore_interop_supported(self)->int:
        """External timeline semaphore interop is supported on the device"""
        ...

    @property
    def memory_pools_supported(self)->int:
        """Device supports using the ::cuMemAllocAsync and ::cuMemPool family of APIs"""
        ...

    @property
    def gpu_direct_rdma_supported(self)->int:
        """Device supports GPUDirect RDMA APIs, like nvidia_p2p_get_pages (see https://docs.nvidia.com/cuda/gpudirect-rdma for more information)"""
        ...

    @property
    def gpu_direct_rdma_flush_writes_options(self)->int:
        """The returned attribute shall be interpreted as a bitmask, where the individual bits are described by the ::CUflushGPUDirectRDMAWritesOptions enum"""
        ...

    @property
    def gpu_direct_rdma_writes_ordering(self)->int:
        """GPUDirect RDMA writes to the device do not need to be flushed for consumers within the scope indicated by the returned attribute. See ::CUGPUDirectRDMAWritesOrdering for the numerical values returned here."""
        ...

    @property
    def mempool_supported_handle_types(self)->int:
        """Handle types supported with mempool based IPC"""
        ...

    @property
    def cluster_launch(self)->int:
        """Indicates device supports cluster launch"""
        ...

    @property
    def deferred_mapping_cuda_array_supported(self)->int:
        """Device supports deferred mapping CUDA arrays and CUDA mipmapped arrays"""
        ...

    @property
    def can_use_64_bit_stream_mem_ops(self)->int:
        """64-bit operations are supported in ::cuStreamBatchMemOp and related MemOp APIs."""
        ...

    @property
    def can_use_stream_wait_value_nor(self)->int:
        """::CU_STREAM_WAIT_VALUE_NOR is supported by MemOp APIs."""
        ...

    @property
    def dma_buf_supported(self)->int:
        """Device supports buffer sharing with dma_buf mechanism. """
        ...

    @property
    def ipc_event_supported(self)->int:
        """Device supports IPC Events. """
        ...

    @property
    def mem_sync_domain_count(self)->int:
        """Number of memory domains the device supports."""
        ...

    @property
    def tensor_map_access_supported(self)->int:
        """Device supports accessing memory using Tensor Map."""
        ...

    @property
    def handle_type_fabric_supported(self)->int:
        """Device supports exporting memory to a fabric handle with cuMemExportToShareableHandle() or requested with cuMemCreate()"""
        ...

    @property
    def unified_function_pointers(self)->int:
        """Device supports unified function pointers."""
        ...

    @property
    def numa_config(self)->int:
        """NUMA configuration of a device: value is of type ::CUdeviceNumaConfig enum"""
        ...

    @property
    def numa_id(self)->int:
        """NUMA node ID of the GPU memory"""
        ...

    @property
    def multicast_supported(self)->int:
        """Device supports switch multicast and reduction operations."""
        ...

    @property
    def mps_enabled(self)->int:
        """Indicates if contexts created on this device will be shared via MPS"""
        ...

    @property
    def host_numa_id(self)->int:
        """NUMA ID of the host node closest to the device. Returns -1 when system does not support NUMA."""
        ...

    @property
    def d3d12_cig_supported(self)->int:
        """Device supports CIG with D3D12."""
        ...

