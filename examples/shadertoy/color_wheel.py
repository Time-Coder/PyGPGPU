from pygf import color3f, float2, float3, cos
from shadertoy_viewer import ShadertoyViewer


class ColorWheel(ShadertoyViewer):

    def mainImage(self, frag_coord: float2) -> color3f:
        # Normalized pixel coordinates (from 0 to 1)
        uv: float2 = frag_coord / self.i_resolution.xy

        # Time varying pixel color
        return 0.5 + 0.5*cos(self.i_time + uv.xyx + float3(0, 2, 4))


if __name__ == "__main__":
    ColorWheel().show()
