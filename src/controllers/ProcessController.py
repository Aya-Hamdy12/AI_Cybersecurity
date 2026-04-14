from .BaseController import BaseController
from .ProjectController import ProjectController
import os
import json
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
        
        sample_df = pd.read_csv(file_path, nrows=2)
        
        # Check if the CSV has headers
        needs_headers = False
        try:
            # If the column name can be converted to a float, it's raw data (no header)
            float(sample_df.columns[0])
            needs_headers = True 
        except ValueError:
            # If it throws a ValueError, the column name is a string (has header)
            needs_headers = False

        if needs_headers:
            feature_path = os.path.join(self.base_dir, "assets", "feature_names.json")
            try:
                with open(feature_path, "r") as f:
                    feature_names = json.load(f)
                
                df = pd.read_csv(file_path, names=feature_names)
            except FileNotFoundError:
                print("Warning: feature_names.json not found. Reading with default columns.")
                df = pd.read_csv(file_path)
        else:
            df = pd.read_csv(file_path)

        # Process the rows into chunks
        chunks = []
        for idx, row in df.iterrows():
            label = str(row.get("label", "unknown"))
            binary_label = str(row.get("binary_label", "unknown"))
            dst_port = str(row.get("dst_port", "unknown"))
            protocol = str(row.get("protocol", "unknown"))

            # Build a Semantic Narrative for historical RAG retrieval
            chunk_text = f"Network Traffic Flow Record (Row Index: {idx})\n"
            chunk_text += f"Threat Classification: {label}\n"
            chunk_text += f"Is Anomaly: {binary_label}\n"
            chunk_text += f"Destination Port: {dst_port}, Protocol: {protocol}\n"
            chunk_text += "="*40 + "\n"
            
            chunk_text += "Raw Network Flow Features:\n"
            features = []
            for col, val in row.items():
                if col not in ['label', 'binary_label']:
                    features.append(f"{col}: {val}")
            
            chunk_text += " | ".join(features)

            # Enhanced metadata for filtering
            chunk = Document(
                page_content=chunk_text,
                metadata={
                    "source": file_path,
                    "row_index": idx,
                    "label": label,
                    "binary_label": binary_label,
                    "dst_port": dst_port 
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





