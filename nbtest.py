from end2end.doc2vec import train_doc2vec
from end2end.seq2seq import Seq2SeqModel
from end2end.preprocess import preprocess_language_model_data

def train_test():
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
    ''' Test the 
    Report:
        Test passed. ~2mins
    '''
    train_doc2vec()

