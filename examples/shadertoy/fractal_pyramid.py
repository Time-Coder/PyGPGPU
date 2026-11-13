from pygpgpu import length, mix, normalize, cross, dot, sign, sin
from pygf import color3f, float2, float3, cos
from shadertoy_viewer import ShadertoyViewer


class FractalPyramid(ShadertoyViewer):
    """The "Fractal Pyramid" raymarcher: a kaleidoscopic fold stepped along a ray.

    The shapes the original spells in GLSL map one-to-one: `rotate` is two floats
    and a manual 2x2 apply (a mat2 would arrive as a structure), `map` folds the
    point eight times and answers the box-ish field `dot(sign(p), p) / 5`,
    `rm` marches 64 steps and accumulates the palette of the distance travelled,
    and `mainImage` builds the camera basis from the orbiting origin. The devices
    carry every piece natively: unary minus and abs expand per component, the
    swizzle augassign stores through its own accessors, `break` sits inside a real
    loop, and mix/normalize/cross/dot/sign take vectors on every backend.
    """

    def rotate(self, p: float2, a: float) -> float2:
        c: float = cos(a)
        s: float = sin(a)
        # p*mat2(c,s,-s,c) spells out as the two dot products the matrix means.
        return float2(p.x * c + p.y * s, -p.x * s + p.y * c)

    def palette(self, d: float) -> float3:
        return mix(float3(0.2, 0.7, 0.9), float3(1.0, 0.0, 1.0), d)

    def map(self, p: float3) -> float:
        for i in range(8):
            t: float = self.i_time * 0.2
            p.xz = self.rotate(p.xz, t)
            p.xy = self.rotate(p.xy, t * 1.89)
            p.xz = abs(p.xz)
            p.xz -= 0.5
        return dot(sign(p), p) / 5.0

    def rm(self, ro: float3, rd: float3) -> float3:
        t: float = 0.0
        col: float3 = float3(0.0)
        for i in range(64):
            p: float3 = ro + rd * t
            d: float = self.map(p) * 0.5
            if d < 0.02:
                break
            if d > 100.0:
                break
            col += self.palette(length(p) * 0.1) / (400.0 * d)
            t += d
        return col

    def mainImage(self, frag_coord: float2) -> color3f:
        uv: float2 = (frag_coord - self.i_resolution.xy / 2.0) / self.i_resolution.x
        ro: float3 = float3(0.0, 0.0, -50.0)
        ro.xz = self.rotate(ro.xz, self.i_time)
        cf: float3 = normalize(-ro)
        cs: float3 = normalize(cross(cf, float3(0.0, 1.0, 0.0)))
        cu: float3 = normalize(cross(cf, cs))
        uuv: float3 = ro + cf * 3.0 + uv.x * cs + uv.y * cu
        rd: float3 = normalize(uuv - ro)
        return self.rm(ro, rd)


if __name__ == "__main__":
    FractalPyramid().show()
