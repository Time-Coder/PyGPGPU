"""Light Cycle -- the Shadertoy of the Week, ported from its GLSL to the device subset.

Original: "Light Cycle" by Noztol on shadertoy.com, the site's featured shader when this
example was written. The port keeps the GLSL's structure one-to-one: every GLSL function is a
python function or method with the same name, every uniform is a ``self`` attribute, and the
camera flies down the same endless square tunnel -- cyan ribs and rings on the walls, orange
streams in the corners, an anti-aliased grid that fades with distance, cone-traced coverage
for the light bars and accumulated volumetric glow for the air between them.

Two spellings differ from the GLSL, both forced by the device subset rather than chosen:

* there is no ``mat2``, so ``rot`` takes the vector it rotates and answers the rotated one;
* ``abs``/``min``/``max``/``smoothstep`` on vectors are refused today (see
  ``tests/known_failures/``), so the helpers expand that arithmetic per component -- the maths
  is unchanged, and scalar spellings stay exactly as the GLSL wrote them.
"""

import math

import numpy as np

from pygpgpu import clamp, config, cross, dot, fract, mix, normalize
from pygf import color3f, float2, float3

from shadertoy_viewer import ShadertoyViewer

config.print_info = False
# The viewer's own import has already chosen GLSL's float32 for this folder; restating it
# keeps the file honest when it is read on its own, and a raymarcher is the one shader
# family that cannot afford the 1:64 fp64 rate a double would cost.
config.type_map = {float: np.float32}

MAX_STEPS: int = 110
MAX_DIST: float = 80.0
SURF_DIST: float = 0.001


def _abs2(v: float2) -> float2:
    """GLSL ``abs(vec2)`` -- component-wise; a vector argument is refused today."""
    return float2(abs(v.x), abs(v.y))


def _abs3(v: float3) -> float3:
    """GLSL ``abs(vec3)``."""
    return float3(abs(v.x), abs(v.y), abs(v.z))


def _max3(v: float3, s: float) -> float3:
    """GLSL ``max(vec3, float)`` -- the component-wise maximum against one scalar."""
    return float3(max(v.x, s), max(v.y, s), max(v.z, s))


def _rot2(v: float2, a: float) -> float2:
    """GLSL ``v * rot(a)`` for the column-major ``mat2(c, -s, s, c)``.

    The GLSL multiplies the vector on the LEFT of the matrix (``p.xy *= rot(..)``), which is
    the row-vector form: ``(c*x - s*y, s*x + c*y)`` -- a plain counter-clockwise turn. The
    matrix-times-vector spelling would turn the other way, and the difference shows: the
    tunnel's roll and the grid's normal un-rotation must use the same direction.
    """
    s = math.sin(a)
    c = math.cos(a)
    return float2(c * v.x - s * v.y, s * v.x + c * v.y)


def _smoothstep1(e0: float, e1: float, x: float) -> float:
    """GLSL ``smoothstep`` for scalars (the device cannot walk pygf's own today)."""
    t = clamp((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _smoothstep3(e0: float, e1: float, v: float3) -> float3:
    """GLSL ``smoothstep(e0, e1, vec3)`` -- component-wise."""
    return float3(_smoothstep1(e0, e1, v.x), _smoothstep1(e0, e1, v.y),
                  _smoothstep1(e0, e1, v.z))


def _sd_box(p: float3, b: float3) -> float:
    """The box SDF: ``length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0)``."""
    qx = abs(p.x) - b.x
    qy = abs(p.y) - b.y
    qz = abs(p.z) - b.z
    ox = max(qx, 0.0)
    oy = max(qy, 0.0)
    oz = max(qz, 0.0)
    outside = math.sqrt(ox * ox + oy * oy + oz * oz)
    inside = min(max(qx, max(qy, qz)), 0.0)
    return outside + inside


def _sd_box2d(p: float2, b: float2) -> float:
    """The 2D box SDF the light bars use: flat against the tunnel wall."""
    qx = abs(p.x) - b.x
    qy = abs(p.y) - b.y
    outside = math.sqrt(max(qx, 0.0) * max(qx, 0.0) + max(qy, 0.0) * max(qy, 0.0))
    inside = min(max(qx, qy), 0.0)
    return outside + inside


def _fold_square(p: float2) -> float2:
    """GLSL foldSquare: mirror into the first quadrant, keeping the greater coordinate first."""
    p = _abs2(p)
    if p.y > p.x:
        p.xy = p.yx
    return p


def _box_filtered_line(coord: float, scale: float, line_width: float, footprint: float) -> float:
    """A box-filtered 1D line grid: covers the pixel footprint instead of point-sampling."""
    w = max(footprint * scale, 0.0001)
    p = coord * scale
    a = p + 0.5 * w
    b = p - 0.5 * w
    coverage = (math.floor(a) + min(fract(a) / line_width, 1.0)
                - math.floor(b) - min(fract(b) / line_width, 1.0)) / (w / line_width)
    return clamp(coverage, 0.0, 1.0)


class LightCycle(ShadertoyViewer):
    """The tunnel. ``mainImage`` is the GLSL's, line for line; the rest are its helpers."""

    def __init__(self, width: int = 640, height: int = 360,
                 title: str = "light cycle") -> None:
        super().__init__(width=width, height=height, title=title)
        # The GLSL's const vec3 globals, as uniforms: vector members ride the receiver.
        self.col_bg = float3(0.004, 0.007, 0.014)
        self.col_cyan = float3(0.0, 0.85, 1.0)
        self.col_orange = float3(1.0, 0.28, 0.0)
        self.col_blue = float3(0.04, 0.18, 0.75)

    # -- the scene ---------------------------------------------------------

    def get_path(self, z: float) -> float2:
        """The tunnel's curving centre line at depth ``z`` (GLSL ``getPath``)."""
        return float2(
            math.sin(z * 0.05) * 2.4 + math.cos(z * 0.025) * 1.0,
            math.cos(z * 0.04) * 2.0 + math.sin(z * 0.02) * 0.8
        )

    def to_tunnel_space(self, p_world: float3) -> float3:
        """World point -> tunnel space: re-centre on the path, then roll with the flight."""
        g = self.get_path(p_world.z)
        rotated = _rot2(float2(p_world.x - g.x, p_world.y - g.y),
                        p_world.z * 0.02 + self.i_time * 0.05)
        return float3(rotated.x, rotated.y, p_world.z)

    def map(self, p_world: float3) -> float3:
        """Scene distance, then the cyan and orange light-bar distances (GLSL ``map``)."""
        p = self.to_tunnel_space(p_world)
        f = _fold_square(float2(p.x, p.y))
        q = float3(f.x, f.y, p.z)

        z_period = 4.0
        # python's % is floored exactly like GLSL mod; the operand stays positive here.
        q_cell_z = (q.z + z_period * 0.5) % z_period - z_period * 0.5

        # 1. Watertight closed square tunnel: solid walls at q.x = 4.0 ...
        d_walls = 4.0 - q.x

        # Carve 0.5-unit deep recessed architectural bays (from q.x = 4.0 to 4.5).
        # Slight 0.012 bevel on cutouts prevents single-pixel normal chatter on edges.
        d_recess_center = _sd_box(float3(q.x - 4.0, q.y - 0.0, q_cell_z - 0.0),
                                  float3(0.5, 1.488, 1.488)) - 0.012
        d_recess_side = _sd_box(float3(q.x - 4.0, q.y - 2.75, q_cell_z - 0.0),
                                float3(0.5, 0.738, 1.488)) - 0.012

        d_carve = min(d_recess_center, d_recess_side)
        d_arch = max(d_walls, -d_carve)

        # 2. Flush 360-degree cyan light ribs.
        d_cyan_rail1 = _sd_box2d(float2(q.x - 4.0, q.y - 1.5), float2(0.05, 0.05))
        d_cyan_rail2 = _sd_box2d(float2(q.x - 4.0, q.y - 2.0), float2(0.05, 0.05))
        d_cyan_rail3 = _sd_box2d(float2(q.x - 4.0, q.y - 3.5), float2(0.05, 0.05))
        d_cyan_rails = min(d_cyan_rail1, min(d_cyan_rail2, d_cyan_rail3))

        d_cyan_ring1 = _sd_box(float3(q.x - 4.0, q.y - 2.0, abs(q_cell_z) - 1.5),
                               float3(0.05, 2.0, 0.05))
        d_cyan_ring2 = _sd_box(float3(q.x - 4.0, q.y - 2.0, abs(q_cell_z) - 2.0),
                               float3(0.05, 2.0, 0.05))
        d_cyan_rings = min(d_cyan_ring1, d_cyan_ring2)

        d_cyan = min(d_cyan_rails, d_cyan_rings)

        # 3. Corner orange light streams.
        d_orange = _sd_box2d(float2(q.x - 3.92, q.y - 3.92), float2(0.11, 0.11))

        # Combine geometry into a single smooth surface so rays never experience
        # jagged material-ID silhouette jumps.
        d_scene = min(d_arch, min(d_cyan, d_orange))

        return float3(d_scene, d_cyan, d_orange)

    def get_normal(self, p: float3) -> float3:
        """The tetrahedron four-tap normal: the GLSL k.xyy / k.yyx / k.yxy / k.xxx trick."""
        h = 0.002
        a = self.map(p + float3(h, -h, -h)).x
        b = self.map(p + float3(-h, -h, h)).x
        c = self.map(p + float3(-h, h, -h)).x
        d = self.map(p + float3(h, h, h)).x
        return normalize(float3(a - b - c + d, -a - b + c + d, -a + b - c + d))

    def get_anti_aliased_grid(self, p_world: float3, n_world: float3, rd: float3,
                              t: float, pixel_spread: float) -> float:
        """The wall grid, box-filtered by the ray cone so it fades cleanly with distance."""
        p = self.to_tunnel_space(p_world)
        f = _fold_square(float2(p.x, p.y))
        q = float3(f.x, f.y, p.z)

        n_local = n_world
        rotated = _rot2(float2(n_local.x, n_local.y), p.z * 0.02 + self.i_time * 0.05)
        n_local = float3(rotated.x, rotated.y, n_local.z)
        n_abs = _abs3(n_local)
        if abs(p.y) > abs(p.x):
            n_abs = float3(n_abs.y, n_abs.x, n_abs.z)

        s_axis = _smoothstep3(0.35, 0.65, n_abs)
        axis_weight = float3(1.0 - s_axis.x, 1.0 - s_axis.y, 1.0 - s_axis.z)

        cos_theta = max(abs(dot(n_world, rd)), 0.15)
        world_footprint = (t * pixel_spread * 1.35) / cos_theta

        gx_minor = _box_filtered_line(q.x + 0.25, 2.0, 0.035, world_footprint) * axis_weight.x
        gy_minor = _box_filtered_line(q.y + 0.25, 2.0, 0.035, world_footprint) * axis_weight.y
        gz_minor = _box_filtered_line(q.z + 0.25, 2.0, 0.035, world_footprint) * axis_weight.z
        line_minor = max(gx_minor, max(gy_minor, gz_minor))

        gy_major = _box_filtered_line(q.y + 0.5, 1.0, 0.032, world_footprint) * axis_weight.y
        gz_major = _box_filtered_line(q.z + 0.5, 1.0, 0.032, world_footprint) * axis_weight.z
        line_major = max(gy_major, gz_major)

        fade_major = math.exp(-0.0007 * t * t)
        fade_minor = math.exp(-0.0016 * t * t)

        c = line_major * 0.85 * fade_major + line_minor * 0.45 * fade_minor

        pulse = math.sin(p.z * 0.35 - self.i_time * 3.0) * 0.5 + 0.5
        c = c * (0.45 + 0.55 * math.pow(pulse, 3.0))

        return clamp(c, 0.0, 1.0)

    # -- the hook ----------------------------------------------------------

    def mainImage(self, frag_coord: float2) -> color3f:
        """One camera ray through the tunnel: the GLSL ``mainImage``, line for line."""
        min_res = min(self.i_resolution.x, self.i_resolution.y)
        uv = (frag_coord - self.i_resolution * 0.5) * (1.0 / min_res)
        pixel_spread = 1.0 / min_res

        uv = uv * (1.0 + dot(uv, uv) * 0.05)

        speed = 8.0
        cam_z = self.i_time * speed
        origin = self.get_path(cam_z)
        ro = float3(origin.x, origin.y, cam_z)

        target_z = cam_z + 5.0
        target = self.get_path(target_z)
        ta = float3(target.x, target.y, target_z)

        fwd = normalize(ta - ro)
        bank_angle = -fwd.x * 0.45 + math.sin(self.i_time * 0.3) * 0.1
        up0 = float3(math.sin(bank_angle), math.cos(bank_angle), 0.0)
        right = normalize(cross(fwd, up0))
        up = cross(right, fwd)

        rd = normalize(fwd * 0.9 + right * uv.x + up * uv.y)

        t = 0.0
        glow = float3(0.0, 0.0, 0.0)

        min_cyan_cone = 1e5
        min_orange_cone = 1e5
        hit_z = 0.0

        # The march: GLSL breaks out of its for loop; the same state is carried in the
        # while condition, with the hit distance kept apart from the stepped one.
        d = 1.0
        steps = 0
        t_hit = MAX_DIST
        while t < MAX_DIST and steps < MAX_STEPS and d >= SURF_DIST:
            p = ro + rd * t
            res = self.map(p)
            d = res.x
            d_cyan = res.y
            d_orange = res.z

            cone_radius = max(t * pixel_spread * 1.25, 0.003)

            min_cyan_cone = min(min_cyan_cone, d_cyan / cone_radius)
            min_orange_cone = min(min_orange_cone, d_orange / cone_radius)

            step_dist = max(d * 0.65, 0.002)

            glow = glow + self.col_cyan * (step_dist * 0.022 / (0.015 + d_cyan * d_cyan * 3.5))

            packet = math.pow(math.sin(p.z * 0.25 + self.i_time * 10.0) * 0.5 + 0.5, 2.5)
            orange_emit = mix(self.col_orange * 0.45, self.col_orange * 2.2, packet)
            glow = glow + orange_emit * (step_dist * 0.035 / (0.012 + d_orange * d_orange * 3.5))

            glow = glow + self.col_blue * (step_dist * 0.0035 / (0.2 + d * d))

            if d < SURF_DIST:
                t_hit = t
                hit_z = p.z

            t = t + step_dist
            steps += 1

        col = self.col_bg

        if t_hit < MAX_DIST:
            p = ro + rd * t_hit
            n = self.get_normal(p)
            v = rd * -1.0

            fresnel = math.pow(clamp(1.0 - max(dot(n, v), 0.0), 0.0, 1.0), 3.0)

            grid = self.get_anti_aliased_grid(p, n, rd, t_hit, pixel_spread)
            base_color = float3(0.008, 0.012, 0.022)
            rim_color = mix(self.col_blue, self.col_cyan, 0.55) * (fresnel * 0.9)
            grid_color = self.col_cyan * (grid * 1.5)
            col = base_color + rim_color + grid_color

            cyan_coverage = 1.0 - _smoothstep1(-0.5, 1.2, min_cyan_cone)
            cyan_surf = self.col_cyan * 2.4 + float3(0.75, 0.75, 0.75)
            col = mix(col, cyan_surf, cyan_coverage)

            orange_coverage = 1.0 - _smoothstep1(-0.5, 1.2, min_orange_cone)
            packet = math.pow(math.sin(hit_z * 0.25 + self.i_time * 10.0) * 0.5 + 0.5, 2.5)
            orange_surf = mix(self.col_orange * 1.0,
                              self.col_orange * 2.8 + float3(0.95, 0.65, 0.25), packet)
            col = mix(col, orange_surf, orange_coverage)

            fog = 1.0 - math.exp(-0.00075 * t_hit * t_hit)
            col = mix(col, self.col_bg, fog)

        col = col + glow * 0.8

        # ACES-ish tonemap, spelled per component (the GLSL writes it as vec3 arithmetic).
        col = float3(
            (col.x * (2.51 * col.x + 0.03)) / (col.x * (2.43 * col.x + 0.59) + 0.14),
            (col.y * (2.51 * col.y + 0.03)) / (col.y * (2.43 * col.y + 0.59) + 0.14),
            (col.z * (2.51 * col.z + 0.03)) / (col.z * (2.43 * col.z + 0.59) + 0.14)
        )

        q = frag_coord / self.i_resolution
        vig = math.pow(16.0 * q.x * q.y * (1.0 - q.x) * (1.0 - q.y), 0.15)
        col = col * mix(0.65, 1.0, vig)

        inv22 = 1.0 / 2.2
        col = float3(
            math.pow(max(col.x, 0.0), inv22),
            math.pow(max(col.y, 0.0), inv22),
            math.pow(max(col.z, 0.0), inv22)
        )

        dither = fract(math.sin(dot(frag_coord, float2(12.9898, 78.233))) * 43758.5453)
        dither = (dither - 0.5) / 255.0
        col = col + float3(dither, dither, dither)

        return color3f(col.x, col.y, col.z)


if __name__ == "__main__":
    LightCycle().show()
