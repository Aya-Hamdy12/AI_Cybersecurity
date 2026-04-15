from pydantic import BaseModel, Field, field_validator, ConfigDict, BeforeValidator, PlainSerializer
from typing import Annotated, Optional
from bson.objectid import ObjectId
from helpers.utils import PyObjectId


class DataChunk(BaseModel):
    id: Optional[PyObjectId] = Field(None, alias="_id")
    chunk_project_id: PyObjectId = Field(...)
    chunk_order: int = Field(..., ge=0)
    chunk_text: str = Field(..., min_length=1)
    chunk_metadata: dict = Field(default_factory=dict)
    

    @field_validator("chunk_text")
    @classmethod
    def validate_chunk_text(cls, value):
        if not value.strip():
            raise ValueError("chunk_text cannot be empty or whitespace")
        return value
    

    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True
    )


    @classmethod
    def get_indexes(cls):
        return [
            {
                "key": [("chunk_project_id", 1)],
                "name": "chunk_project_id_index_1",
                "unique": False
            }
        ]