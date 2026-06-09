"""Thin wrapper around Chakra's protobuf schema.

We import directly from ASTRA-sim's vendored Chakra and re-export the
classes / constants we need. Centralizing this here means the rest of
the package never reaches into ASTRA-sim's submodule paths.
"""

from __future__ import annotations

import sys

# ASTRA-sim ships Chakra under extern/graph_frontend/chakra. Add its
# schema + utils dirs to sys.path so generated stubs resolve.
_ASTRA_CHAKRA = os.environ.get("ASTRA_CHAKRA", "/opt/astra-sim/extern/graph_frontend/chakra")
for _p in (
    f"{_ASTRA_CHAKRA}/schema/protobuf",
    f"{_ASTRA_CHAKRA}/src/third_party/utils",
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Re-export Chakra types under stable names.
from et_def_pb2 import (   # noqa: E402
    GlobalMetadata,
    Node as ChakraNode,
    AttributeProto as ChakraAttr,
    NodeType,
    COMP_NODE,
    COMM_SEND_NODE,
    COMM_RECV_NODE,
    COMM_COLL_NODE,
)
from et_def_pb2 import (   # noqa: E402
    ALL_REDUCE,
    ALL_GATHER,
    REDUCE_SCATTER,
    BROADCAST,
    CollectiveCommType,
)
from protolib import (     # noqa: E402
    encodeMessage as encode_message,
)


__all__ = [
    "GlobalMetadata", "ChakraNode", "ChakraAttr", "NodeType",
    "COMP_NODE", "COMM_SEND_NODE", "COMM_RECV_NODE", "COMM_COLL_NODE",
    "ALL_REDUCE", "ALL_GATHER", "REDUCE_SCATTER", "BROADCAST",
    "CollectiveCommType",
    "encode_message",
]


# ----- typed attribute factories --------------------------------------------
# Most node-attr setting in ASTRA-sim takes a name + a value of one specific
# proto field. These helpers spell out the field so call sites stay readable.

def attr_int64(name: str, value: int) -> ChakraAttr:
    a = ChakraAttr(name=name)
    a.int64_val = int(value)
    return a


def attr_uint64(name: str, value: int) -> ChakraAttr:
    a = ChakraAttr(name=name)
    a.uint64_val = int(value)
    return a


def attr_int32(name: str, value: int) -> ChakraAttr:
    a = ChakraAttr(name=name)
    a.int32_val = int(value)
    return a


def attr_bool(name: str, value: bool) -> ChakraAttr:
    a = ChakraAttr(name=name)
    a.bool_val = bool(value)
    return a


def attr_string(name: str, value: str) -> ChakraAttr:
    a = ChakraAttr(name=name)
    a.string_val = str(value)
    return a
