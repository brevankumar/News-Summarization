import boto3
import os

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

for root, dirs, files in os.walk(downloads_dir): 
    for filename in files: 
        print(filename)