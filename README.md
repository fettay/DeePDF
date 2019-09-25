# Detecting malicious PDF using CNN 

### Description

This repository contains the code accompanying the paper: Detecting malicious PDF using CNN. It is implemented using PyTorch. 

### Setup

To set up just create a virtual environment with **python3** and run:

    pip install -r requirements.txt


### Prerequisites

In order to train a model, you need to have:
 
- The pdfs files you want to train on in a local folder 
- A csv containing about information about the files. The format should be the same as samples.csv

*To require access to the malicious files used in the paper, please contact one of the authors.*

### Run a training ###

To run a training, run the file *train.py*

###### Example:

    python3 train.py ModelB samples.csv data/pdfs/ --name training1 --gpu cuda:3

###### It saves:

- The model in the folder trainings/. It should be loaded with torch.load().
- The logs in the folder logs/.
- A PNG file containing the ROC on the test set in the current working directory.

###### Usage:

    usage: train.py [-h] [--name NAME] [--gpu GPU] [--resample]
                    model files_csv data_path

    positional arguments:
    model        Model to use, should be either 'ModelA', 'ModelB', or 'ModelC'
    files_csv    CSV containing the files for the training and some info. Format
                should be the same as sample.csv
    data_path    Directory in which the files are stored, the name of the files
                must be to the hash in the csv file.

    optional arguments:
    -h, --help   show this help message and exit
    --name NAME  Name of the training (for the log file, the model object and
                the ROC picture)
    --gpu GPU    Which GPU to use, default will be cuda:0
    --resample   Whether to resample the train set