from .BaseController import BaseController
from .ProjectController import ProjectController
import os
from langchain_community.document_loaders import TextLoader, PyMuPDFLoader, CSVLoader, JSONLoader
from models import ProcessingEnum
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd
from langchain.schema import Document

class ProcessController(BaseController):

    def __init__(self, project_id: str):
        super().__init__()

        self.project_id = project_id
        self.project_path = ProjectController().get_project_path(project_id=project_id)
    
    def get_file_extension(self, file_id: str):
        return os.path.splitext(file_id)[-1]

    def get_file_loader(self, file_id: str):
        file_extension = self.get_file_extension(file_id=file_id)

        file_path = os.path.join(self.project_path, file_id)

        if file_extension == ProcessingEnum.TXT.value:
            return TextLoader(file_path, encoding='utf-8')
        elif file_extension == ProcessingEnum.CSV.value:
            return CSVLoader(file_path, encoding='utf-8')
        elif file_extension == ProcessingEnum.PDF.value:
            return PyMuPDFLoader(file_path)
        elif file_extension == ProcessingEnum.JSON.value:
            return JSONLoader(file_path, jq_schema='.[]', text_content=False)

        else:
            return None



    def get_file_content(self, file_id: str):
        loader = self.get_file_loader(file_id=file_id)

        if loader is None:
            return None

        documents = loader.load()

        return documents
    
    def process_csv_by_row(self, file_id: str):
        file_path = os.path.join(self.project_path, file_id)
        df = pd.read_csv(file_path)

        chunks = []
        for idx, row in df.iterrows():
            row_text = "\n".join([
                f"{col}: {val}" 
                for col, val in row.items()
            ])

            chunk = Document(
                page_content=row_text,
                metadata={
                    "source": file_path,
                    "row_index": idx,
                    "label": str(row.get("Label", "unknown")),
                    "binary_label": str(row.get("binary_label", "unknown"))
                }
            )
            chunks.append(chunk)

        return chunks


    def process_file_content(self, file_content: list, file_id: str,
                            chunk_size: int=100, overlap_size: int=20):
        file_extension = self.get_file_extension(file_id=file_id)
        if not file_content or len(file_content) == 0:
            return None
        
        if file_extension == ProcessingEnum.CSV.value:
            return self.process_csv_by_row(file_id)
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=overlap_size,
            length_function=len
        )
        
        file_content_texts = [
            rec.page_content 
            for rec in file_content
        ]

        file_content_metadata =[
            rec.metadata
            for rec in file_content
        ]

        chunks = text_splitter.create_documents(
            file_content_texts, 
            metadatas=file_content_metadata
        )

        return chunks





