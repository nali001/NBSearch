# This script is dedicated to test modules in the NBSearch project 
import os
from end2end.doc2vec import train_doc2vec
from end2end.seq2seqv2 import Seq2SeqModel
from end2end.preprocessv2 import preprocessing, preprocess_language_model_data

# Global parameters
model_option = 'bilstm'
out_path = 'testdata' 

def train_test():
    ''' Test the traing process of translation models. 
    The amount of epochs is set to be 1 to save time. 
    '''
    model = Seq2SeqModel(out_path, model_option)
    model.create_model()
    model.train_model(batch_size=120, epochs=10)
    # model.evaluate_seq2seq_model(nums=0,model_option=option)
    # model.evaluate_seq2seq_model(nums=2)

    
    model = Seq2SeqModel(out_path, model_option)

    # Where is final_comments? 
    model.predict_seq2seq_model(out_path)

def predict_test():
    model = Seq2SeqModel(out_path, model_option)
    if os.path.isfile(os.getcwd()+'/' + model_option + '_seq2seq_model.h5'):
        model.predict_seq2seq_model(out_path)

def evaluate_test():
    pass

def doc2vec_test():
    ''' Test the language models. 
    Report:
        Test passed. ~2mins
    '''
    train_doc2vec()

def preprocessing_test():
    ''' Test the preprocessing part. 
    Report: 

    '''
    path = 'testdata/notebooks/'
    out_path = 'testdata'
    preprocessing(path, out_path)


