
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
from keras.utils.np_utils import to_categorical

class CharRNN:
    def __init__(self, sess, input_dim, output_dim, embedding_size, rnn_cell_size, num_layers, end_elem, sess_chkpt=''):
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rnn_cell_size = rnn_cell_size
        self.num_layers = num_layers
        self.state_shape =  num_layers*2*rnn_cell_size
        self.end_elem = end_elem
        
        # Main architecrure
        
        self.X = tf.placeholder(tf.float64, shape=[None, None, input_dim])
        
        self.state = tf.placeholder(tf.float64, shape=[None, self.state_shape])
        
        self.lengths = tf.placeholder(tf.float64, shape=[None])
        
        # Embedding
        n_steps = tf.shape(self.X)[1]
        
        flat_X = tf.reshape(self.X, [-1, input_dim])
        
        embedding_matrix = tf.get_variable('Embedding_Matrix', shape=[input_dim, embedding_size], 
                                               initializer=xavier_initializer(), dtype=tf.float64)
        
        embedded = tf.reshape(tf.nn.relu(tf.matmul(flat_X, embedding_matrix)), [-1, n_steps, embedding_size])
        
        # RNN
        rnn_cells = [tf.contrib.rnn.LSTMCell(rnn_cell_size, state_is_tuple=False, initializer=xavier_initializer()) 
                                 for i in range(num_layers)]

        multiple_cells = tf.contrib.rnn.MultiRNNCell(rnn_cells, state_is_tuple=False)

        rnn_output, self.new_state = tf.nn.dynamic_rnn(multiple_cells, embedded, initial_state=self.state, 
                                                       dtype=tf.float64, sequence_length=self.lengths)
        
        # First dense
        
        f_d = tf.layers.dense(rnn_output, rnn_cell_size//2, activation=tf.nn.relu, kernel_initializer=xavier_initializer())
        
        # Output
        
        output_logits = tf.layers.dense(f_d, output_dim, kernel_initializer=xavier_initializer())

        self.output = tf.nn.softmax(output_logits)
        
        # Training utils
        
        self.Y = tf.placeholder(tf.float64, shape=[None, None, output_dim])

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_logits, labels=self.Y))
        
        # Optimizer 
        
        self.optim = tf.train.AdamOptimizer().minimize(self.loss)
        
        # Saver
        
        self.saver = tf.train.Saver()
        
        # Stuff
        
        self.sess = sess
        
        if len(sess_chkpt):
            self.saver.restore(self.sess, sess_chkpt)
        else:
            self.sess.run(tf.global_variables_initializer())

    def train_step(self, X, Y, lengths):
        
        initial_states = np.zeros([len(X), self.state_shape])
        
        feed_dict = {
            self.X: X,
            self.Y: Y,
            self.state: initial_states,
            self.lengths: lengths
        }
        
        _, loss = self.sess.run([self.optim, self.loss], feed_dict=feed_dict)
        
        return loss
    
    def generate_element(self, x, zero_state=True):
        
        if zero_state:
            state = np.zeros([1, self.state_shape])
        else:
            state = self.previous_state
        
        feed_dict = {
            self.X: [x],
            self.state: state,
            self.lengths: [len(x)]
        }
        
        next_element, new_state = self.sess.run([self.output, self.new_state], feed_dict=feed_dict)
        
        self.previous_state = new_state
        
        return next_element
    
    def make_choice(self, probs, rand):
        if rand:
            return np.random.choice(np.arange(self.output_dim), p=probs)
        return np.argmax(probs)
    
    def generate_sequence(self, start, max_len=np.inf, rand=True):
        
        num_features = self.input_dim - self.output_dim
        
        features = start[0][:num_features]
        
        sequence = [np.argmax(x[num_features:]) for x in start]

        probs = self.generate_element(start, zero_state=True)
        
        next_element = self.make_choice(probs[0][0], rand)

        while next_element != self.end_elem and len(sequence) < max_len:
        
            sequence.append(next_element)
            
            next_input = np.concatenate([features, to_categorical(next_element, self.output_dim)[0]])
            
            probs = self.generate_element([next_input], False)
            
            next_element = self.make_choice(probs[0][0], rand)
            
        return sequence[1:]
    
    def save(self, path):
        self.saver.save(self.sess, path)