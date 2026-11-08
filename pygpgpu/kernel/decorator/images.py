# Image arguments: the address check and the sampler.
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...image import Image



def _check_image_address(image:object, backend:str)->None:
    """Refuse an address mode this backend cannot honour.

    Only CLAMP is portable. WRAP and MIRROR are defined and implemented in the host
    (python-backend) sampler, but the device samplers read the edge texel instead of
    wrapping: with texel-space coordinates the drivers ignore both CUDA's
    CU_TR_ADDRESS_MODE_WRAP/_MIRROR and OpenCL's CL_ADDRESS_REPEAT/_MIRRORED_REPEAT,
    which was measured on every GPU available while developing this. A launch is refused
    rather than silently returning numbers that differ from the python backend.
    """
    from ...image import ImageAddress

    if image.address is ImageAddress.CLAMP:
        return
    raise ValueError(
        "the %s backend cannot honour address=%r: with texel-space (not normalized) "
        "coordinates the device samplers ignore the wrap and mirror modes and read the "
        "edge texel instead, so this image would not match the python backend. Use "
        "ImageAddress.CLAMP on a GPU backend, or run the kernel on the python backend, "
        "which implements all three modes."
        % (backend, image.address.value))


def _image_sampler(image:"Image")->object:
    """The OpenCL sampler matching an Image's sampling rules.

    OpenCL takes the sampler as a kernel argument while CUDA bakes it into the
    texture object, so this is where the backend-neutral filter/address enums turn
    into an ``sampler_t``. The enum names are the neutral ones, and the mapping here is
    where they become OpenCL's spellings (NEAREST for point, CLAMP_TO_EDGE for clamp).
    """
    from ...image import ImageAddress, ImageFilter
    from ...opencl.driver import (
        cl_addressing_mode,
        cl_filter_mode,
        sampler_t,
    )

    _check_image_address(image, "opencl")

    filter_mode = {
        ImageFilter.POINT: cl_filter_mode.CL_FILTER_NEAREST,
        ImageFilter.LINEAR: cl_filter_mode.CL_FILTER_LINEAR,
    }[image.filter]
    address_mode = {
        ImageAddress.CLAMP: cl_addressing_mode.CL_ADDRESS_CLAMP_TO_EDGE,
    }[image.address]

    # Coordinates are texel-space, matching CUDA and the host path.
    return sampler_t(normalized_coords=False,
                     addressing_mode=address_mode,
                     filter_mode=filter_mode)
