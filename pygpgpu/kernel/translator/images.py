from __future__ import annotations
import ast
from typing import (List)
from .notation import (
    Translated,
)
from .predicates import (
    image_channels,
    image_scalar,
    is_int,
)


class _ImagesMixin:

    def _reject_metal_image(self, node:ast.expr, writing:bool)->None:
        """Refuse an image on the Metal backend with the reason, not a compile error.

        MSL *does* have texture objects (`texture2d<float, access::sample>` and
        `access::write`), but the whole image path is Metal's own and is not
        implemented: the parameter type, the sampling call with its sampler state,
        the store call, and the host-side `MTLTexture` binding all differ from the
        OpenCL formulation this backend reuses. Without them the generated source
        carries `__read_only image2d_t`, which MSL has no notion of, so the compiler
        would fail on an undeclared type. The translator is the right place to say
        so, because the message can name the backends that do work.
        """
        if not self._metal_syntax:
            return
        action = "write" if writing else "read"
        self._error(
            "the metal backend does not support images (%s) yet: the MSL texture "
            "formulation (texture2d<float, access::sample|write>, the sampling call "
            "with its sampler state, and the host-side MTLTexture binding) is not "
            "implemented, so the generated source would name an image2d_t that MSL "
            "does not have. Use the cuda, opencl or python backend for image kernels."
            % action, node)


    def _reject_hip_image(self, node:ast.expr, writing:bool)->None:
        """Refuse an image on the HIP backend with the reason, not a compile error.

        HIP reuses CUDA code generation, so an image parameter is emitted as a
        texture/surface pair and the device would fail to compile. Beyond that, the
        device may have no texture hardware at all: on the tested ROCm 7.2 Windows
        build with an integrated Radeon, hipTexObjectCreate reports "Texture not
        supported on the device". Either way a clear message here is better than a
        compiler diagnostic about an undeclared type.
        """
        if self.launch_backend != "hip":
            return
        action = "write" if writing else "read"
        self._error(
            "the hip backend does not support images (%s): HIP reuses CUDA code "
            "generation, which has no texture object type, and hipTexObjectCreate is "
            "rejected with \"operation not supported\" on hardware without texture "
            "support. Use the cuda, opencl or python backend for image kernels."
            % action, node)


    def _emit_image_read(self, image_code:str, image_t:object, node:ast.Subscript)->Translated:
        """Translate ``img[x, y]`` into the sampling call of the current backend.

        Coordinates are texel space, as :mod:`pygpgpu.image` documents: an integer
        coordinate names a texel and samples its centre, while a fractional one
        addresses a position between texels. CUDA expresses both with a float
        coordinate, OpenCL needs int2 for the former and float2 for the latter, so
        the coordinate type decides which form is emitted.
        """
        self._reject_metal_image(node, writing=False)
        self._reject_hip_image(node, writing=False)
        channels:int = image_channels(image_t)
        scalar:str = image_scalar(image_t)

        indices = self._image_indices(node)
        if channels not in (1, 4):
            self._error(
                "sampling an image of %d channels is not supported: use 1 or 4"
                % channels, node)

        if self._opencl_syntax:
            # OpenCL takes a float coordinate for interpolation, and an integer
            # coordinate means the texel boundary rather than its centre -- which a
            # linear sampler blends across, unlike CUDA where an integer subscript
            # lands on the centre. Both integer and fractional subscripts are
            # therefore emitted as float coordinates, with integers moved to +0.5 so
            # every backend samples the same position.
            parts = []
            for code, t in indices:
                if is_int(t):
                    parts.append("((float)(%s) + 0.5f)" % code)
                else:
                    parts.append("((float)(%s))" % code)
            coord_code = "((float%d)(%s))" % (len(parts), ", ".join(parts))
            sampler = self._image_sampler(image_code)
            call = "read_image%s(%s, %s, %s)" % (
                {"float": "f", "int": "i", "uint": "ui"}[scalar],
                image_code, sampler, coord_code)
        else:
            # CUDA: a float coordinate; integer subscripts land on the texel centre.
            parts = []
            for code, t in indices:
                if is_int(t):
                    parts.append("((float)(%s) + 0.5f)" % code)
                else:
                    parts.append("((float)(%s))" % code)
            if scalar == "float":
                call = "tex%dD<float>(%s, %s)" % (
                    len(parts), image_code, ", ".join(parts))
            else:
                self._error(
                    "sampling a %s image is not supported on cuda: only float images "
                    "can be sampled" % scalar, node)

        if channels == 4:
            if self._opencl_syntax:
                return "(%s).xyzw" % call, self._image_vec_type(scalar, 4)
            return call, self._image_vec_type(scalar, 4)
        if self._opencl_syntax:
            # read_imagef returns a vector; single-channel takes .x
            return "(%s).x" % call, scalar
        return call, scalar


    def _image_indices(self, node:ast.Subscript)->List[Translated]:
        """Compile an image subscript into (x, y, z) order.

        A kernel writes ``img[y, x]`` because that mirrors the host array's row-major
        shape, while every sampling call takes (x, y) -- so the compiled coordinates
        are reversed here, exactly like an array subscript's flattening reverses them.
        """
        index:ast.expr = node.slice
        items = list(index.elts) if isinstance(index, ast.Tuple) else [index]
        compiled = [self._visit_expression(item) for item in items]
        return list(reversed(compiled))


    def _image_vec_type(self, scalar:str, channels:int)->str:
        return "%s%d" % (scalar, channels)


    def _image_param_type(self, name:str, image_t:object)->str:
        """The declaration type of an image parameter on the current backend.

        CUDA reads through a texture object and writes through a surface object, so
        the parameter's type follows how the kernel uses it. OpenCL uses one image
        type for both directions, with a qualifier saying which.
        """
        if self._opencl_syntax:
            written:bool = name in self.required_image_surfaces
            qualifier:str = "__write_only" if written else "__read_only"
            return f"{qualifier} image2d_t"

        if name in self.required_image_surfaces:
            return "cudaSurfaceObject_t"
        return "cudaTextureObject_t"


    def _emit_image_write(self, image_code:str, image_t:object,
                          target:ast.Subscript, value_code:str)->None:
        """Translate ``img[x, y] = v`` into the store call of the current backend.

        A write names one texel, so the coordinate is always integral here: a
        fractional subscript is rejected rather than silently rounded, because there
        is no filtering rule for a store.
        """
        self._reject_metal_image(target, writing=True)
        self._reject_hip_image(target, writing=True)
        channels:int = image_channels(image_t)
        scalar:str = image_scalar(image_t)

        # Record the parameter as a write target: the parameter's declared type
        # depends on it, and the host side needs a surface rather than a texture.
        self.required_image_surfaces.add(image_code)

        indices = self._image_indices(target)
        for _, t in indices:
            if not is_int(t):
                self._error(
                    "an image store needs integer coordinates: a fractional "
                    "coordinate has no texel to write", target)

        if self._opencl_syntax:
            coord_code = "((int%d)(%s))" % (len(indices),
                                            ", ".join(c for c, _ in indices))
            if channels == 1:
                packed = "((float4)(%s))" % value_code
            else:
                packed = "((float4)(%s))" % value_code
            self._emit("write_image%s(%s, %s, %s);" % (
                {"float": "f", "int": "i", "uint": "ui"}[scalar],
                image_code, coord_code, packed))
            return

        # CUDA: a surface store. Its first axis is addressed in bytes, so the x
        # coordinate is scaled by the element size.
        coords = [c for c, _ in indices]
        element_bytes = 4
        x_code = "((%s) * %d)" % (coords[0], element_bytes)
        rest = coords[1:]
        if channels == 1:
            self._emit("surf%dDwrite<%s>(%s, %s, %s);" % (
                len(coords), scalar, value_code, image_code,
                ", ".join([x_code] + rest)))
        else:
            self._emit("surf%dDwrite<vec%d>(%s, %s, %s);" % (
                len(coords), channels, value_code, image_code,
                ", ".join([x_code] + rest)))


    def _surface_name(self, image_code:str)->str:
        return "__surface_for_%s" % image_code


    def _image_sampler(self, image_code:str)->str:
        """The sampler argument an OpenCL image read needs.

        The sampler is fixed per image, so it is emitted as a constant at translation
        time rather than passed as another kernel parameter -- that keeps the kernel
        signature identical to the CUDA one.
        """
        self.required_image_samplers.add(image_code)
        return self._sampler_name(image_code)


    def _sampler_name(self, image_code:str)->str:
        return "__sampler_for_%s" % image_code
