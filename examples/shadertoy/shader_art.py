import math
from pygf import color3f, cos, float2, float3, fract, length
from shadertoy_viewer import ShadertoyViewer


class ShaderArt(ShadertoyViewer):

    def mainImage(self, frag_coord: float2) -> color3f:
        uv: float2 = (2.0 * frag_coord - self.i_resolution.xy) / self.i_resolution.y
        uv0: float2 = uv
        final_color: color3f = color3f(0.0)

        for i in range(4):
            uv = fract(uv * 1.5) - 0.5
            d: float = length(uv) * math.exp(-length(uv0))
            col: float3 = self.palette(length(uv0) + i*.4 + self.i_time*.4)
            d = math.sin(d*8. + self.i_time)/8.
            d = abs(d)
            d = pow(0.01/d, 1.2)
            final_color += col * d

        return final_color

    @staticmethod
    def palette(t: float):
        a: float3 = float3(0.5, 0.5, 0.5)
        b: float3 = float3(0.5, 0.5, 0.5)
        c: float3 = float3(1.0, 1.0, 1.0)
        d: float3 = float3(0.263,0.416,0.557)

        return a + b*cos(6.28318*(c*t+d))


if __name__ == "__main__":
    ShaderArt().show()
