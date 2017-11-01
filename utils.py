import numpy as np
from keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm_notebook

def to_one_hot(smile, append=True):
    
    seq = list(smile)
    
    if append:
        seq.append(end_elem)
    
    indexes = list(map(lambda x: elems.index(x), seq))
    
    return to_categorical(indexes, len(elems))


def to_smile(seq):
    
    new_seq = seq
    
    for i, el in enumerate(seq):
        if el == len(elems)-1:
            new_seq = seq[:i]
            break
        
    return ''.join(map(lambda x: elems[x], new_seq))

def prepare_seq(seq, features):
    
    input_raw = to_categorical(seq[:-1], len(elems))
    
    input = np.array([np.concatenate([features, x]) for x in input_raw])
    
    outputs = to_categorical(seq[1:], len(elems))
    
    return input, outputs

def iterate_minibatches(seqs, features, batchsize, shuffle=False):
    
    assert len(seqs) == len(features)
    
    if shuffle:
        indices = np.arange(len(seqs))
        np.random.shuffle(indices)
        
    for start_idx in range(0, len(seqs) - batchsize + 1, batchsize):
        
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        
        lengths = list(map(lambda x: len(x)-1, seqs[excerpt]))
        
        our_seqs = pad_sequences(seqs[excerpt], padding='post')
        
        X, Y = zip(*map(lambda x: prepare_seq(x[0], x[1]), zip(*[our_seqs, features[excerpt]])))
        
        yield np.array(X), np.array(Y), np.array(lengths)
        
def generate_new(model, features):
    
    start_vec = to_categorical(elems.index(start_elem), len(elems))[0]
    
    raw = model.generate_sequence([np.concatenate([features, start_vec])], rand=True)
    
    return to_smile(raw)

        
start_elem = '<Start>'        
end_elem = '<END>'

elems = [start_elem, '#', '%', '(', ')', '+', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '=', '@', 'A', 'B', 'C', 'F', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'S', 'T', 'V', 'Z', '[', '\\', ']', 'a', 'c', 'e', 'g', 'i', 'l', 'n', 'o', 'p', 'r', 's', 't', 'u', end_elem]