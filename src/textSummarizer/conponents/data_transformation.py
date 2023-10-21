import os
from textSummarizer.logging import logger
from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk
from textSummarizer.entity import DataTransformationConfig
import boto3
import sagemaker
import botocore
from datasets.filesystems import S3FileSystem



class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)


    
    def convert_examples_to_features(self,example_batch):
        input_encodings = self.tokenizer(example_batch['text'] , max_length = 1024, truncation = True )
        
        with self.tokenizer.as_target_tokenizer():
            target_encodings = self.tokenizer(example_batch['summary'], max_length = 128, truncation = True )
            
        return {
            'input_ids' : input_encodings['input_ids'],
            'attention_mask': input_encodings['attention_mask'],
            'labels': target_encodings['input_ids']
        }
    

    def convert(self):
        dataset_newsroom = load_dataset(self.config.data_path)
        dataset_newsroom_pt = dataset_newsroom.map(self.convert_examples_to_features, batched = True)
        dataset_newsroom_pt.save_to_disk(os.path.join(self.config.root_dir,"dataset"))

        """# Load the dataset
        train_dataset, test_dataset = load_dataset(self.config.data_path, split=['train', 'test'])

        # tokenize dataset
        train_dataset = train_dataset.map(self.convert_examples_to_features, batched=True)
        test_dataset = test_dataset.map(self.convert_examples_to_features, batched=True)

        sm_boto3 = boto3.client("sagemaker") 

        sess= sagemaker.Session()

        region = sess.boto_session.region_name 

        bucket = 'summarization.nlp' 

        role = sagemaker.get_execution_role()

        s3_prefix = 'transformed_data'

        s3 = S3FileSystem()  

        # save train_dataset to s3
        training_input_path = f's3://{bucket}/{s3_prefix}/train'
        train_dataset.save_to_disk(training_input_path, fs=s3)

        # save test_dataset to s3
        test_input_path = f's3://{bucket}/{s3_prefix}/test'
        test_dataset.save_to_disk(test_input_path, fs=s3)"""