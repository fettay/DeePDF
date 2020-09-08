from data_helpers import get_datasets
from torch.utils.data import DataLoader
from torch.nn.modules.module import _addindent
import torch
import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import time
import logging
from uuid import uuid4
from bmodels import INPUT_LENGTH, INPUT_HEIGHT, BModelA
import argparse
import pickle
import hashlib
from utils import torch_summarize, PDFDataSet, predict_bayes, find_detection_at
from sklearn.metrics import roc_curve
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import os


VALID_RATIO = 0.2
BATCH_SIZE = 64
NB_EPOCHS = 50

display_step = 15
test_step = 450
MODEL_DIC = {'modela': BModelA}


def elbo(out, y, kl, beta):
    loss = F.binary_cross_entropy_with_logits(out, y)
    return loss + beta * kl



    
def run(model, training_csv, data_path, training_id=None, gpu=None, resample=False, cont=False):
    
    model = model.lower()
    
    if training_id is None:
        training_id = str(uuid4())

    if not os.path.exists('trainings'):
        os.makedirs('trainings')
    
    if not os.path.exists('logs'):
        os.makedirs('logs')

    model_file = 'trainings/%s' % training_id

    logfile = 'logs/%s.out' % training_id
    logger = logging.getLogger()
    logger.warning('Writing logs in : %s' % logfile)

    logging.basicConfig(level=logging.DEBUG, filename=logfile)
    logger.debug('Training id: %s' % training_id)
    
    # Preprocessing
    if gpu is None:
        gpu = "cuda:0"
    device = torch.device(gpu if torch.cuda.is_available() else "cpu")
    logger.debug('Used device %s' % device)
    df_train, df_valid, df_test = get_datasets(training_csv)
    
    if resample:
        df_train = df_train.sample(frac=1, replace=True)
        logger.debug('Resample is on')
    else:
        logger.debug('Resample is off')

    learning_rate = 5e-3
        
    
    logger.debug('Train size %d, Valid size %d,  Test size %d' % (df_train.shape[0],
                                                               df_valid.shape[0], df_test.shape[0]))
    training_sha256 = hashlib.sha256(pickle.dumps(df_train)).hexdigest()
    logger.debug('Train df sha256: %s' % training_sha256)
    
    dataloader = DataLoader(PDFDataSet(df_train, data_path), batch_size=BATCH_SIZE, shuffle=False)
    validloader = DataLoader(PDFDataSet(df_valid, data_path), batch_size=BATCH_SIZE, shuffle=False)
    
    if not cont:
        model_cls = MODEL_DIC[model]
        model = model_cls()
    else:
        model = torch.load(model_file , map_location={'cuda:%d' % i: gpu for i in range(8)})
        model.eval()

    adam_optim = optim.Adam([{'params':model.parameters()}],lr=learning_rate)
    
    logger.debug(torch_summarize(model))

    model = model.to(device)


    ### TRAINING ###

    step_msg = 'Step:{} | Loss:{:.6f} | Acc:{:.4f} | Time:{:.2f}'
    valid_msg = 'Val_detection:{:.4f}'
    history = {}
    history['tr_loss'] = []
    history['tr_acc'] = []

    logger.debug('step,tr_loss, tr_acc, val_loss, val_acc, time\n')

    best_1_percent = 0.0
    total_step = 0
    step_cost_time = 0
    max_step = (int(df_train.shape[0] / BATCH_SIZE) + 1) * NB_EPOCHS   # Last number is the number of epochs
        
    while total_step < max_step:

        ### TRAINING ### 
        for step,batch_data in enumerate(dataloader):
            start = time.time()

            adam_optim.zero_grad()

            cur_batch_size = batch_data[0].size(0)

            inp = batch_data[0].to(device)
            inp = Variable(inp.long(),requires_grad=False)
            label = batch_data[1].to(device)
            label = Variable(label.float(),requires_grad=False)
            pred, kl = model(inp)
        
            beta = 1 / df_train.shape[0]
            loss = elbo(pred, label, kl, beta)
            loss.backward()
            adam_optim.step()
            history['tr_loss'].append(loss.cpu().data.item())
            predicted_bool = (torch.sigmoid(pred) > .5)[0, :].float().cpu().data.numpy()
            history['tr_acc'].extend(list(label.cpu().data.numpy().astype(int) == predicted_bool.astype(int)))

            step_cost_time = time.time()-start

            if step % display_step == 0:
                logger.debug(step_msg.format(total_step,np.mean(history['tr_loss']),
                                      np.mean(history['tr_acc']),step_cost_time))
                history['tr_loss'] = []
                history['tr_acc'] = []

            ### VALIDATION ###    

            if total_step % test_step == 0:

                pred, labels = predict_bayes(model, validloader, device, verbose=False)
                roc = roc_curve(labels, pred)
                _, detection_1_percent = find_detection_at(roc, .01)

                logger.info(valid_msg.format(detection_1_percent))

                if best_1_percent < detection_1_percent:
                    best_1_percent = detection_1_percent
                    torch.save(model, model_file)
                    logger.debug('Model saved at %s' % model_file)
            
            ### END VALIDATION ###

            total_step += 1
    ### END TRAINING ###


    ### TESTING ###
    testloader = DataLoader(PDFDataSet(df_valid, data_path), batch_size=BATCH_SIZE, shuffle=False)
    test_pred, test_labels = predict(model, testloader, device, verbose=False)
    fpr, tpr, _ = roc_curve(test_labels, test_pred)
    plt.plot(fpr, tpr, label="ROC on training")
    plt.savefig(training_id + '_roc.png')
    logger.info("The ROC curve on test has been stored in %s_roc.png. Training is done" % training_id)

    
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, default='ModelB', help="Model to use, should be either 'ModelA', 'ModelB', or 'ModelC'")
    parser.add_argument('files_csv', type=str, default='sample.csv', help="CSV containing the files for the training and some info. Format should be the same as sample.csv")
    parser.add_argument('data_path', type=str, default='data/', help="Directory in which the files are stored, the name of the files must be to the hash in the csv file. ")
    parser.add_argument('--name', type=str, default=None, help="Name of the training (for the log file, the model object and the ROC picture)")
    parser.add_argument('--gpu', type=str, default=None, help="Which GPU to use, default will be cuda:0")
    parser.add_argument('--resample', action='store_true', help="Whether to resample the train set")
    parser.add_argument('--cont', action='store_true', help="Whether to continue old training")
    args = parser.parse_args()
    run(args.model.lower(), args.files_csv, args.data_path, args.name, args.gpu, args.resample, args.cont)        
