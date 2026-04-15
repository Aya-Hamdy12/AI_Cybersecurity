from enum import Enum

class ResponseSignal(Enum):

    FILE_VALIDATION_SUCCESS = "File validate successfully"
    FILE_UPLOAD_SUCCESS = "File upload success"
    FILE_UPLOAD_FAILED = "File upload failed!"
    FILE_TYPE_NOT_SUPPORTED = "Invalid file type"
    FILE_SIZE_EXCEEDED = "File size exceeds the maximum limit!"
    PROCESSING_SUCCESS = "File processed successfully"
    PROCESSING_FAILED = "File processing failed!"
    NO_FILES_ERROR = "Not found any files to process!"
    FILE_ID_ERROR = "No file found with this ID"


