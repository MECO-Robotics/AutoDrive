"""2026 REBUILT field geometry in a field-relative meter coordinate frame."""
from __future__ import annotations

from dataclasses import dataclass
import math


INCH = 0.0254
ALLIANCE_ZONE_DEPTH = 158.6 * INCH
# REBUILT bump nominal envelope. The 0.5-inch HDPE top contributes the initial
# step; the remaining rise over half the footprint yields the nominal 15° ramp.
BUMP_HEIGHT = 6.513 * INCH
BUMP_PANEL_THICKNESS = 0.5 * INCH
BUMP_RAMP_RISE = BUMP_HEIGHT - BUMP_PANEL_THICKNESS
BUMP_RAMP_GRADE = BUMP_RAMP_RISE / (44.4 * INCH / 2)
BUMP_ROLLING_RESISTANCE = 0.025
BUMP_ROBOT_CG_HEIGHT = 0.30
# AD* keeps a modest conservative traversal cost in addition to gravity/load
# effects simulated from the ramp profile.
BUMP_SPEED_SCALE = 0.88
BUMP_ACCELERATION_SCALE = 0.75


def midfield_respawn_points(count: int, field_length: float,
                            field_width: float) -> list[tuple[float, float]]:
    """Return stable spawn points within the central neutral-pile footprint."""
    count = max(0, int(count))
    if count == 0:
        return []
    # Scored fuel returns to the central neutral pile (72 x 206 in), rather
    # than appearing anywhere across the full midfield corridor.
    center_x, center_y = float(field_length) / 2, float(field_width) / 2
    half_depth, half_length = 0.915, 2.615
    x_min, x_max = center_x - half_depth, center_x + half_depth
    y_min, y_max = center_y - half_length, center_y + half_length
    span_x, span_y = x_max - x_min, y_max - y_min
    points = []
    for index in range(count):
        # A deterministic low-discrepancy scatter avoids a visible grid while
        # keeping the lookup table stable across resets and devices.
        x = x_min + span_x * ((index * 0.6180339887498949) % 1.0)
        y = y_min + span_y * ((index * 0.7548776662466927) % 1.0)
        points.append((x, y))
    return points


@dataclass(frozen=True)
class FieldBox:
    name: str
    x: float
    y: float
    length: float
    width: float
    color: str = "structure"

    def as_tensor(self) -> tuple[float, float, float, float]:
        return self.x, self.y, self.length / 2, self.width / 2

    def as_dict(self) -> dict[str, float | str]:
        return {"name": self.name, "x": self.x, "y": self.y,
                "length": self.length, "width": self.width, "color": self.color}


def rebuilt_field(length: float = 651.22 * INCH,
                  width: float = 317.7 * INCH) -> tuple[FieldBox, ...]:
    """Return REBUILT structure and terrain footprints.

    Coordinates start at the red alliance wall and run downfield along x;
    y=0 is the lower long guardrail. BUMP boxes are traversable rough terrain
    and TRENCH boxes are open floor beneath an overhead arm. Other boxes are
    treated as solid footprint-level structures in the planar model.
    """
    center_y = width / 2
    # The alliance-zone boundary meets the hub's near face.
    hub_x = ALLIANCE_ZONE_DEPTH + 47.0 * INCH / 2
    hub_size = 47.0 * INCH
    bump_depth, bump_width = 44.4 * INCH, 73.0 * INCH
    bump_y_offset = (hub_size + bump_width) / 2
    boxes = [
        FieldBox("red_hub", hub_x, center_y, hub_size, hub_size, "hub-red"),
        FieldBox("blue_hub", length - hub_x, center_y, hub_size, hub_size, "hub-blue"),
    ]
    for alliance, x in (("red", hub_x), ("blue", length - hub_x)):
        boxes.extend((
            FieldBox(f"{alliance}_bump_lower", x, center_y - bump_y_offset,
                     bump_depth, bump_width, f"bump-{alliance}"),
            FieldBox(f"{alliance}_bump_upper", x, center_y + bump_y_offset,
                     bump_depth, bump_width, f"bump-{alliance}"),
        ))
    tower_length, tower_width = 45.0 * INCH, 49.25 * INCH
    tower_x = tower_length / 2
    boxes.extend((
        FieldBox("red_tower", tower_x, center_y, tower_length, tower_width, "tower-red"),
        FieldBox("blue_tower", length - tower_x, center_y, tower_length,
                 tower_width, "tower-blue"),
    ))
    # Only the two tall uprights are ground-level tower obstacles. They sit at
    # the field-facing end of the tower base, 32.25in apart across its width.
    upright_thickness, upright_depth = 1.5 * INCH, 3.5 * INCH
    upright_spacing = 32.25 * INCH
    upright_x = tower_length - upright_depth / 2
    for alliance, x in (("red", upright_x), ("blue", length - upright_x)):
        for side, y in (("lower", center_y - upright_spacing / 2),
                        ("upper", center_y + upright_spacing / 2)):
            boxes.append(FieldBox(f"{alliance}_tower_upright_{side}", x, y,
                                  upright_depth, upright_thickness,
                                  f"tower-{alliance}"))
    # Arms run inward from the long guardrails until they meet the bumps.
    # Their 65.65in span is across the field; the 47in depth runs downfield.
    trench_x = hub_x
    trench_depth, trench_width = 47.0 * INCH, 65.65 * INCH
    trench_clear_width = 50.34 * INCH
    # The measured clearance runs from the guardrail-side edge to the in-field
    # support. The remaining width is one support strip at the in-field end.
    trench_support_width = trench_width - trench_clear_width
    trench_y = trench_width / 2
    for alliance, x in (("red", trench_x), ("blue", length - trench_x)):
        boxes.extend((
            FieldBox(f"{alliance}_trench_lower", x, trench_y, trench_depth,
                     trench_width, f"trench-{alliance}"),
            FieldBox(f"{alliance}_trench_upper", x, width - trench_y,
                     trench_depth, trench_width, f"trench-{alliance}"),
        ))
        # The outer edge of each trench arm meets the field perimeter. Do not
        # create a second ground obstacle there: the guardrail already defines
        # the field boundary. Only the in-field supports bound the 50.34in
        # opening and block robots.
        support_centers = (
            ("lower", trench_width - trench_support_width / 2),
            ("upper", width - trench_width + trench_support_width / 2),
        )
        for side, support_y in support_centers:
            boxes.append(FieldBox(f"{alliance}_trench_support_{side}",
                x, support_y, trench_depth, trench_support_width, f"support-{alliance}"))
    depot_depth, depot_width = 27.0 * INCH, 42.0 * INCH
    depot_x = depot_depth / 2
    depot_y_offset = 94.0 * INCH
    boxes.extend((
        FieldBox("red_depot", depot_x, center_y - depot_y_offset,
                 depot_depth, depot_width, "depot-red"),
        FieldBox("blue_depot", length - depot_x, center_y + depot_y_offset,
                 depot_depth, depot_width, "depot-blue"),
    ))
    return tuple(box for box in boxes if 0 <= box.y <= width)


def static_collision_boxes(boxes: tuple[FieldBox, ...]) -> tuple[FieldBox, ...]:
    """Return ground-blocking structures, excluding driveable field zones."""
    return tuple(box for box in boxes
                 if not box.name.endswith(("_depot", "_tower")) and
                 (not box.name.startswith(("red_bump", "blue_bump",
                                            "red_trench", "blue_trench")) or
                  "_trench_support_" in box.name))


def bump_boxes(boxes: tuple[FieldBox, ...]) -> tuple[FieldBox, ...]:
    return tuple(box for box in boxes if "_bump_" in box.name)


def box_observation_radius(box: FieldBox) -> float:
    """Signed radial feature: positive for solid obstacles, negative for bumps."""
    radius = math.hypot(box.length, box.width) / 2
    return -radius if "_bump_" in box.name else radius
