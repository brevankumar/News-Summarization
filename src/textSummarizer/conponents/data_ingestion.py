import os
import urllib.request as request
import zipfile
from textSummarizer.logging import logger
from textSummarizer.utils.common import get_size
from textSummarizer.entity import DataIngestionConfig
from pathlib import Path
import boto3
import os




class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config


    
    def download_file(self):
        
        aws_access_key_id = 'AKIASU2C6O4I6EX34RZO'
        aws_secret_access_key = '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'
        aws_region = 'us-east-1'

        client = boto3.client('s3', aws_access_key_id=aws_access_key_id, 
                                    aws_secret_access_key=aws_secret_access_key, 
                                    region_name=aws_region)

        bucket = 'rev.nlpdata'

        cur_path = os.getcwd()

        file= 'dataset.zip' 

        filename = os.path.join(cur_path, 'artifacts\Data_ingestion', file)

        client.download_file(
                            Bucket = bucket,
                            Key=file,
                            Filename=filename
                            )

        downloads_dir = os.path.join(cur_path,'artifacts\Data_ingestion')

        return downloads_dir

        
    
    def extract_zip_file(self):
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)