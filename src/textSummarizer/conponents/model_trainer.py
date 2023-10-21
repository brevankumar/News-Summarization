from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset, load_from_disk
import torch
import os
from textSummarizer.entity import ModelTrainerConfig

import sagemaker.huggingface
from sagemaker.huggingface import HuggingFace
import sagemaker
import boto3
from datasets.filesystems import S3FileSystem



class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config


"""
    def train(self):
        
    
        sm_boto3 = boto3.client("sagemaker") 

        sess= sagemaker.Session()

        region = sess.boto_session.region_name 

        bucket = 'summarization.nlp' 

        role = sagemaker.get_execution_role()

        s3_prefix = 'transformed_data'

        s3 = S3FileSystem()  
        
        pytorch_version = '1.7'

        python_version  ='py36'
                
        # for Data Parallel training 
        distribution = {"smdistributed": { "dataparallel": { "enabled": True } } }
        

        git_config = {'repo': 'https://github.com/huggingface/transformers.git','branch': 'v4.26.0'}


        # hyperparameters, which are passed into the training job
        hyperparameters={'epochs': 1,                          # number of training epochs
                        'train_batch_size': 32,               # batch size for training
                        'eval_batch_size': 64,                # batch size for evaluation
                        'learning_rate': 3e-5,                # learning rate used during training
                        'model_id':'google/pegasus-cnn_dailymail', # pre-trained model
                        'fp16': True,                         # Whether to use 16-bit (mixed) precision training
                        }

        # configuration for running training on smdistributed Data Parallel
        #distribution = {'smdistributed':{'dataparallel':{ 'enabled': True }}}


        # create the Estimator
        huggingface_estimator = HuggingFace(
            entry_point='run_summarization.py',
            source_dir='./examples/pytorch/summarization',
            git_config=git_config,
            instance_type='ml.p3.16xlarge',
            instance_count=1,
            transformers_version='4.6',
            pytorch_version=pytorch_version,
            py_version=python_version,
            role=role,
            hyperparameters=hyperparameters,
            #distribution=distribution,
        )

        
        # save train_dataset to s3
        training_input_path = f's3://{bucket}/{s3_prefix}/train'

        # save test_dataset to s3
        test_input_path = f's3://{bucket}/{s3_prefix}/test'

        # define a data input dictonary with our uploaded s3 uris
        data = {
            'train': training_input_path,
            'test': test_input_path
        }

        # starting the train job with our uploaded datasets as input
        huggingface_estimator.fit(data, wait=True)



        ## Save model
        # huggingface_estimator.save_pretrained(os.path.join(self.config.root_dir,"pegasus-newsroom-model"))

        ## Save tokenizer
        #tokenizer.save_pretrained(os.path.join(self.config.root_dir,"tokenizer"))

"""


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config


    
    def train(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)
        model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_ckpt).to(device)
        seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model_pegasus)
        
        #loading data 
        dataset_samsum_pt = load_from_disk(self.config.data_path)

        # trainer_args = TrainingArguments(
        #     output_dir=self.config.root_dir, num_train_epochs=self.config.num_train_epochs, warmup_steps=self.config.warmup_steps,
        #     per_device_train_batch_size=self.config.per_device_train_batch_size, per_device_eval_batch_size=self.config.per_device_train_batch_size,
        #     weight_decay=self.config.weight_decay, logging_steps=self.config.logging_steps,
        #     evaluation_strategy=self.config.evaluation_strategy, eval_steps=self.config.eval_steps, save_steps=1e6,
        #     gradient_accumulation_steps=self.config.gradient_accumulation_steps
        # ) 


        trainer_args = TrainingArguments(
            output_dir=self.config.root_dir, num_train_epochs=1, warmup_steps=500,
            per_device_train_batch_size=1, per_device_eval_batch_size=1,
            weight_decay=0.01, logging_steps=10,
            evaluation_strategy='steps', eval_steps=500, save_steps=1e6,
            gradient_accumulation_steps=16
        ) 

        trainer = Trainer(model=model_pegasus, args=trainer_args,
                  tokenizer=tokenizer, data_collator=seq2seq_data_collator,
                  train_dataset=dataset_samsum_pt["test"], 
                  eval_dataset=dataset_samsum_pt["validation"])
        
        trainer.train()

        ## Save model
        model_pegasus.save_pretrained(os.path.join(self.config.root_dir,"pegasus-samsum-model"))
        ## Save tokenizer
        tokenizer.save_pretrained(os.path.join(self.config.root_dir,"tokenizer"))