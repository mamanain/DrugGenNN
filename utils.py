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


def split_seq(seq, split_size=20, stride=5):
    
    seq = to_one_hot(seq)
    
    X = []; Y = []
    
    for i in range(0, len(seq)-split_size, stride):
        
        X.append(seq[i:i+split_size])
        Y.append(seq[i+split_size])
        
    return X, Y



def prepare_seq(seq, feature_vec, split_size=20, stride=5, min=2):
        
    X = []; Y = []
    
    seq = [elems.index(x) for x in seq]
    
    seq.append(elems.index(end_elem))
    
    if len(seq) <= split_size and len(seq) > min:
        return [list(feature_vec) + seq[:-1]], to_categorical(seq[-1], len(elems))
    
    for i in range(0, len(seq)-split_size, stride):
    
        X.extend([list(feature_vec) + seq[i:i+split_size]])
        Y.extend(to_categorical(seq[i+split_size], len(elems)))
        
    return X, Y



def get_train(texts, features, split_size=20, stride=5):
    X, Y = [], []
    
    for text, feature_vec in tqdm_notebook(list(zip(*[texts, features]))):
        
        x_temp, y_temp = prepare_seq(text, feature_vec, split_size, stride)

        X.extend(x_temp)
        Y.extend(y_temp)
        
    return np.array(X), np.array(Y)


def make_trainable(X_batch, seq_index=3):
    
    new_batch = []
    
    features = [x[:seq_index] for x in X_batch]
    
    seqs = pad_sequences([x[seq_index:] for x in X_batch], padding='post', value=elems.index(end_elem))
    
    for feature, seq in zip(*[features, seqs]):
        one_hot = to_categorical(seq, len(elems))
      
        new_batch.append([np.concatenate([feature, x]) for x in one_hot])
    
    return np.array(new_batch)


def iterate_minibatches(X, Y, batchsize, shuffle=False):
    assert len(X) == len(Y)
    if shuffle:
        indices = np.arange(len(X))
        np.random.shuffle(indices)
    for start_idx in range(0, len(X) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield X[excerpt], Y[excerpt]
        
start_elem = '<Start>'        
end_elem = '<END>'

elems = [start_elem, '#', '%', '(', ')', '+', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '=', '@', 'A', 'B', 'C', 'F', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'S', 'T', 'V', 'Z', '[', '\\', ']', 'a', 'c', 'e', 'g', 'i', 'l', 'n', 'o', 'p', 'r', 's', 't', 'u', end_elem]