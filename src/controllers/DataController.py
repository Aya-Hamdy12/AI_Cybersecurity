from .BaseController import BaseController
from fastapi import UploadFile
from models import ResponseSignal
from .ProjectController import ProjectController
import re
import os


class DataController(BaseController):

    def __init__(self):
        super().__init__()

    def validate_uploaded_file(self, file: UploadFile):
        # Implement validation logic for the uploaded file
        if file.content_type not in self.app_settings.FILE_ALLOWED_TYPES:
            return False,ResponseSignal.FILE_TYPE_NOT_SUPPORTED.value
        if file.size > self.app_settings.FILE_MAX_SIZE_MB * 1024 * 1024:
            return False,ResponseSignal.FILE_SIZE_EXCEEDED.value
        
        return True,ResponseSignal.FILE_VALIDATION_SUCCESS.value
    


    def generate_unique_filepath(self, original_filename: str, project_id: str):
        random_key = self.generate_random_string()
        project_path = ProjectController().get_project_path(project_id=project_id)

        clean_filename = self.get_clean_filename(original_filename=original_filename)

        new_file_path = os.path.join(
            project_path, 
            f"{random_key}_{clean_filename}"
        )
        while os.path.exists(new_file_path):
            random_key = self.generate_random_string()
            new_file_path = os.path.join(
                project_path,
                f"{random_key}_{clean_filename}"
            )

        return new_file_path, f"{random_key}_{clean_filename}"

    def get_clean_filename(self, original_filename: str):
        # Remove special characters and spaces from the original filename
        clean_filename = re.sub(r'[^\w.]', '', original_filename.strip())
        clean_filename = clean_filename.replace(" ", "_")

        return clean_filename












