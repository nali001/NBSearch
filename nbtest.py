# This script is dedicated to test modules in the NBSearch project 
from end2end.doc2vec import train_doc2vec
from end2end.seq2seq import Seq2SeqModel
from end2end.preprocessv2 import preprocessing, preprocess_language_model_data

def train_test():
    ''' Test the traing process of translation models. 
    The amount of epochs is set to be 1 to save time. 
    '''
    options = ['bilstm'] #, 'gru', 'lstm', 'lstmattention']
    for option in options:
        model = Seq2SeqModel(model_option=option)
        model.create_model()
        model.train_model(batch_size=120, epochs=1)
        # model.evaluate_seq2seq_model(nums=0,model_option=option)
        model.evaluate_seq2seq_model(nums=2)
    
    
    model = Seq2SeqModel(model_option='bilstm')

    # Where is final_comments? 
    model.predict_seq2seq_model(filename='data/final_comments.csv')

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


