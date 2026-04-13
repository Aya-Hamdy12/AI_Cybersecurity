from pydantic import BaseModel
from typing import Optional



class ProcessRequest(BaseModel):
    file_id: str
    chunk_size: Optional[int] = 100  # Default chunk size of 1MB
    overlap_size: Optional[int] = 20  # Default overlap size of 20KB
    do_reset: Optional[int] = 0  # Whether to reset existing processed data for the file



