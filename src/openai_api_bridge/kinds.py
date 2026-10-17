"""The closed sets of "kind" strings the bridge passes around.

Typed as ``Literal`` so a misspelt kind is a type error rather than a value
that quietly matches nothing. Kept outside ``backends`` because the file store
uses them too, and infrastructure should not import from the backends.
"""

from __future__ import annotations

from typing import Literal

# What a generated, stored asset is. Also the ``kind`` column's CHECK set.
AssetKind = Literal["image", "video"]

# What a listed model does, as surfaced on ``/v1/models``.
ModelKind = Literal["image", "video", "chat", "embedding"]
