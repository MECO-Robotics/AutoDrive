"""Static V4 score-event ABI declarations.

This module declares the fixed event row layout only. It intentionally does
not pack, decode, hash, validate dynamic values, allocate device tensors, or
define unresolved piece-generation, tick, or event-sequence semantics. Dynamic
event production and digest work must remain on the assigned device under the
accepted V4 contract.
"""

SCHEMA_VERSION = 4
EVENT_TABLE_CAPACITY = 2304
EVENT_ROW_BYTES = 59
EVENT_TABLE_BYTES = 135936

# (field name, ABI scalar type, byte width, byte offset)
# The offsets are the packed little-endian V4 order with no implicit padding.
SCORE_EVENT_FIELDS = (
    ("valid", "bool", 1, 0),
    ("rollout_row_index", "uint8", 1, 1),
    ("episode_event_sequence", "uint16", 2, 2),
    ("tick", "int64", 8, 4),
    ("match_elapsed_seconds", "float64", 8, 12),
    ("world_slot", "int64", 8, 20),
    ("alliance", "uint8", 1, 28),
    ("robot_slot", "int64", 8, 29),
    ("piece_slot", "int64", 8, 37),
    ("piece_generation", "int64", 8, 45),
    ("event_type", "uint16", 2, 53),
    ("score_units", "uint16", 2, 55),
    ("source_site_code", "uint16", 2, 57),
)

EVENT_TYPE_FUEL_SCORE_AWARDED = 1
SOURCE_SITE_VECTOR_SCORE_MUTATION = 1
SOURCE_SITE_PER_ROBOT_SCORE_MUTATION = 2
FUEL_SCORE_UNITS_PER_ACCEPTED_MUTATION = 1
