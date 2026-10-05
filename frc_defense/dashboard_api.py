"""Read-only API payload builders used by the dashboard HTTP handler."""
from __future__ import annotations

import json


def field_layout_payload() -> bytes:
    """Return the compact field geometry document served by ``/api/field-layout``."""
    from .field import (
        ALLIANCE_ZONE_DEPTH,
        bump_boxes,
        rebuilt_field,
        static_collision_boxes,
    )

    boxes = rebuilt_field()
    field_length = 651.22 * 0.0254
    field_width = 317.7 * 0.0254
    return json.dumps(
        {
            "length": field_length,
            "width": field_width,
            "alliance_zone_depth": ALLIANCE_ZONE_DEPTH,
            "elements": [box.as_dict() for box in boxes],
            "colliders": [box.as_dict() for box in static_collision_boxes(boxes)],
            "bump_regions": [box.as_dict() for box in bump_boxes(boxes)],
        },
        separators=(",", ":"),
    ).encode()
