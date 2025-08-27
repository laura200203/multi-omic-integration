# config.py
from skopt.space import Integer, Categorical, Real

epochs = [300]

search_spaces = {
    # Original
    'DirectPred': [
        Integer(65.5, 66.5, name='latent_dim'),
        Real(0.472, 0.473, name='hidden_dim_factor'), # relative size of the hidden_dim w.r.t input_dim 
        Real(0.002, 0.0022, prior='log-uniform', name='lr'),
        Integer(36.5, 37.5, name='supervisor_hidden_dim'),
        Categorical(epochs, name='epochs')
    ], 
    #'DirectPred': [
        #Integer(57, 59, name='latent_dim'),
        #Real(0.46, 0.48, name='hidden_dim_factor'), # relative size of the hidden_dim w.r.t input_dim 
        #Real(0.007, 0.01, prior='log-uniform', name='lr'),
        #Integer(22, 25, name='supervisor_hidden_dim'),
        #Categorical(epochs, name='epochs')
    #], 
    #'DirectPred': [
        #Integer(23,36, name='latent_dim'),
        #Real(0.214, 0.215, name='hidden_dim_factor'), # relative size of the hidden_dim w.r.t input_dim 
        #Real(0.0012, 0.0013, prior='log-uniform', name='lr'),
        #Integer(30, 32, name='supervisor_hidden_dim'),
        #Categorical(epochs, name='epochs')
    #], 
    'supervised_vae': [
        Integer(16, 128, name='latent_dim'),
        Real(0.2, 0.5, name='hidden_dim_factor'), # relative size of the hidden_dim w.r.t input_dim 
        Integer(8, 32, name='supervisor_hidden_dim'),
        Real(0.0001, 0.01, prior='log-uniform', name='lr'),
        Categorical(epochs, name='epochs')
    ],
    'CrossModalPred': [
        Integer(16, 128, name='latent_dim'),
        Real(0.2, 0.5, name='hidden_dim_factor'), # relative size of the hidden_dim w.r.t input_dim 
        Integer(8, 32, name='supervisor_hidden_dim'),
        Real(0.0001, 0.01, prior='log-uniform', name='lr'),
        Categorical(epochs, name='epochs')
    ],
    'MultiTripletNetwork': [
        Integer(16, 128, name='latent_dim'),
        Real(0.2, 0.5, name='hidden_dim_factor'), # relative size of the hidden_dim w.r.t input_dim 
        Integer(8, 32, name='supervisor_hidden_dim'),
        Real(0.0001, 0.01, prior='log-uniform', name='lr'),
        Categorical(epochs, name='epochs')
    ],
    'GNN': [
        Integer(16, 128, name='latent_dim'),
        Integer(4, 32, name='node_embedding_dim'), # node embedding dimensions
        Integer(1, 4, name='num_convs'), # number of convolutional layers 
        Real(0.0001, 0.01, prior='log-uniform', name='lr'),
        Integer(8, 32, name='supervisor_hidden_dim'),
        Categorical(epochs, name='epochs'),
        Categorical(['relu'], name="activation")
    ]
}
