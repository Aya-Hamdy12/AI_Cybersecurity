from typing import Annotated
from pydantic import BeforeValidator, PlainSerializer
from bson.objectid import ObjectId

try:
    from typing import TypeAlias
except ImportError:
    from typing_extensions import TypeAlias

PyObjectId: TypeAlias = Annotated[
    ObjectId,
    BeforeValidator(lambda x: ObjectId(x) if isinstance(x, str) and ObjectId.is_valid(x) else x),
    PlainSerializer(lambda x: str(x), return_type=str, when_used="json")
]