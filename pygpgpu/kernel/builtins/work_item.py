from __future__ import annotations


from ..host_runtime import local_extent, work_item, work_size_extent

KERNEL_TRANSLATING: bool = False


def get_global_id(dim: int) -> int:
    """The global id of this work item along ``dim``."""
    return work_item(dim, "get_global_id")


def get_local_id(dim: int) -> int:
    """The id of this work item inside its group along ``dim``."""
    return work_item(dim, "get_local_id") % local_extent(dim, "get_local_id")


def get_group_id(dim: int) -> int:
    """The id of this work item's group along ``dim``."""
    return work_item(dim, "get_group_id") // local_extent(dim, "get_group_id")


def get_global_size(dim: int) -> int:
    """The total number of work items along ``dim``."""
    global_extent, _ = work_size_extent(dim, "get_global_size")
    return global_extent


def get_local_size(dim: int) -> int:
    """The number of work items per group along ``dim``."""
    _, local_size = work_size_extent(dim, "get_local_size")
    return local_size


def get_num_groups(dim: int) -> int:
    """The number of groups along ``dim``."""
    global_extent, local_size = work_size_extent(dim, "get_num_groups")
    return -(-global_extent // local_size)
