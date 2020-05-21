from keras.layers import Lambda, Input, Dense, Activation 
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.optimizers import Adam
from keras.utils import plot_model
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from MidiFile import MIDIFile



class AnnealingCallback(tf.keras.callbacks.Callback):
    def __init__(self, weight, cutoff_epoch):
        self.weight = weight
        self.cutoff_epoch = cutoff_epoch
    def on_epoch_end (self, epoch, logs={}):
        value = 1*epoch/self.cutoff_epoch if (epoch<self.cutoff_epoch) else 1
        K.set_value(self.weight, value)

class VAE():
    def __init__(self, beta, encoder_dims, decoder_dims, latent_dim, decoder_activations,encoder_activations):
        self.load_data()
        self.beta = K.variable(beta)
        self.encoder_dims = encoder_dims
        self.decoder_dims = decoder_dims
        self.latent_dim = latent_dim
        self.input_shape = (14*32,)
        self.decoder_activations = decoder_activations
        self.encoder_activations = encoder_activations
        self.num_encoder_layers = len(encoder_dims)
        self.num_decoder_layers = len(decoder_dims)
        config = tf.compat.v1.ConfigProto(
            gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
        )
        config.gpu_options.allow_growth = True
        session = tf.compat.v1.Session(config=config)
        tf.compat.v1.keras.backend.set_session(session)
        #tf.compat.v1.disable_eager_execution()
        self.build_model()

    def load_data(self):
        x_train0 = np.load('data_matrix0.npy')
        x_train1 = np.load('data_matrix1.npy')
        x_train2 = np.load('data_matrix2.npy')
        x_train3 = np.load('data_matrix3.npy')
        x_train4 = np.load('data_matrix4.npy')
        x_train5 = np.load('data_matrix5.npy')
        self.x_train = np.concatenate((x_train0,x_train1,x_train2,x_train3,x_train4,x_train5))
        self.x_testval = np.load('data_matrix6.npy')
        num_testval = self.x_testval.shape[0]
        self.x_test = self.x_testval[0:num_testval//2,:]
        self.x_val = self.x_testval[num_testval//2:,:]

    def sampling(self,args):
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.shape(z_mean)[1]
        print(batch,dim)
        epsilon = K.random_normal(shape = (batch,dim))
        z_sampled = z_mean + K.exp(0.5*z_log_var)*epsilon   
        
        return z_sampled

    def build_model(self):
        self.inputs = Input(shape = self.input_shape, name = 'encoder_input')
        previous_layer = self.inputs
        for i in range(self.num_encoder_layers):
            x = Dense(self.encoder_dims[i], activation = self.encoder_activations[i])(previous_layer)
            previous_layer = x
            
        self.z_mean = Dense(self.latent_dim, name='z_mean')(previous_layer)
        self.z_log_var = Dense(self.latent_dim, name='z_log_var')(previous_layer)

        self.z = Lambda(self.sampling, output_shape=(self.latent_dim,), name='z')([self.z_mean, self.z_log_var])

        encoder = Model(self.inputs, [self.z_mean, self.z_log_var, self.z], name='encoder')
        encoder.summary()
        plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)

        latent_inputs = Input(shape = (self.latent_dim,), name = 'z_sampling')
        previous_layer = latent_inputs
        for i in range(self.num_decoder_layers):
            x = Dense(self.decoder_dims[i], activation = self.decoder_activations[i])(previous_layer)
            previous_layer = x    
        outputs = Dense(self.input_shape[0], activation = 'sigmoid')(previous_layer)

        self.decoder = Model(latent_inputs, outputs, name='decoder')
        self.decoder.summary()
        plot_model(self.decoder, to_file='vae_mlp_decoder.png', show_shapes=True)

        self.outputs_vae = self.decoder(encoder(self.inputs)[2])
        self.vae = Model(self.inputs, self.outputs_vae, name='vae_mlp')
    def real_loss(self,beta):
        def custom_loss(y_true,y_pred):
            reconstruction_loss = binary_crossentropy(self.inputs, self.outputs_vae)     
            reconstruction_loss *= self.input_shape[0]   
            kl_loss = 1 + self.z_log_var - self.z_mean**2 - K.exp(self.z_log_var)
            kl_loss = K.sum(kl_loss, axis=-1)
            kl_loss *= -0.5*self.beta
            vae_loss = K.mean(reconstruction_loss + kl_loss)
            return vae_loss
        return custom_loss

        self.vae.compile(optimizer=opt)
    def train(self,epochs,batch_size,save,name='', cutoff = 5):
        print("Initializing training...")
        opt = Adam(learning_rate = 0.001)
        self.vae.compile(optimizer = opt, loss = self.real_loss(self.beta))
        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience= 200*cutoff)
        model_fit=self.vae.fit(self.x_train,self.x_train,epochs=epochs,batch_size=batch_size,validation_data=(self.x_val, self.x_val),callbacks = [AnnealingCallback(self.beta,cutoff),es],verbose = 2)
        training_loss_history = model_fit.history["loss"]
        validation_loss_history = model_fit.history["val_loss"]
        training_loss = np.array(training_loss_history)
        validation_loss = np.array(validation_loss_history)
        if save:
            np.savetxt('train_loss_'+name+'.txt', training_loss, delimiter=",")
            np.savetxt('val_loss_'+name+'.txt', validation_loss, delimiter=",")
            self.vae.save_weights(name+'_'+str(epochs)+'e_'+str(batch_size)+'b.h5')
        return validation_loss[-1]
    
    def save_weights(self,name):
        self.vae.save_weights(name+'.h5')

beta = 0
"""
#STANDARD
print("-----------STANDARD-------------")
encoder_dims = [64,32]
decoder_dims = [32,64]
latent_dim = 4
decoder_activations = ['relu']*len(encoder_dims)
encoder_activations = ['relu']*len(decoder_dims)
vae = VAE(beta, encoder_dims, decoder_dims, latent_dim, decoder_activations,encoder_activations)
epochs = 25
batch_size = 128
save = True
name = 'STANDARD'
vae.train(epochs,batch_size,save,name)

#STANDARD2
print("-----------STANDARD2-------------")
encoder_dims = [64,32]
decoder_dims = [32,64]
latent_dim = 20
decoder_activations = ['relu']*len(encoder_dims)
encoder_activations = ['relu']*len(decoder_dims)
vae = VAE(beta, encoder_dims, decoder_dims, latent_dim, decoder_activations,encoder_activations)
epochs = 25
batch_size = 128
save = True
name = 'STANDARD2'
vae.train(epochs,batch_size,save,name)

#big
print("-----------BIG-------------")
encoder_dims = [128,64]
decoder_dims = [64,128]
latent_dim = 4
decoder_activations = ['relu']*len(encoder_dims)
encoder_activations = ['relu']*len(decoder_dims)
vae = VAE(beta, encoder_dims, decoder_dims, latent_dim, decoder_activations,encoder_activations)
epochs = 25
batch_size = 128
save = True
name = 'BIG'
vae.train(epochs,batch_size,save,name)

#big2
print("-----------BIG2-------------")
encoder_dims = [128,64]
decoder_dims = [64,128]
latent_dim = 20
decoder_activations = ['relu']*len(encoder_dims)
encoder_activations = ['relu']*len(decoder_dims)
vae = VAE(beta, encoder_dims, decoder_dims, latent_dim, decoder_activations,encoder_activations)
epochs = 25
batch_size = 128
save = True
name = 'BIG2'
vae.train(epochs,batch_size,save,name)

#large
print("-----------LARGE-------------")
encoder_dims = [256,128]
decoder_dims = [128,256]
latent_dim = 4
decoder_activations = ['relu']*len(encoder_dims)
encoder_activations = ['relu']*len(decoder_dims)
vae = VAE(beta, encoder_dims, decoder_dims, latent_dim, decoder_activations,encoder_activations)
epochs = 25
batch_size = 128
save = True
name = 'LARGE'
vae.train(epochs,batch_size,save,name)

#large2
print("-----------LARGE2-------------")
encoder_dims = [256,128]
decoder_dims = [128,256]
latent_dim = 20
decoder_activations = ['relu']*len(encoder_dims)
encoder_activations = ['relu']*len(decoder_dims)
vae = VAE(beta, encoder_dims, decoder_dims, latent_dim, decoder_activations,encoder_activations)
epochs = 25
batch_size = 128
save = True
name = 'LARGE2'
vae.train(epochs,batch_size,save,name)

#deep
print("-----------DEEP-------------")
encoder_dims = [128,64,32]
decoder_dims = [32,64,128]
latent_dim = 4
decoder_activations = ['relu']*len(encoder_dims)
encoder_activations = ['relu']*len(decoder_dims)
vae = VAE(beta, encoder_dims, decoder_dims, latent_dim, decoder_activations,encoder_activations)
epochs = 25
batch_size = 128
save = True
name = 'DEEP'
vae.train(epochs,batch_size,save,name)

#deep2
print("-----------DEEP2-------------")
encoder_dims = [128,64,32]
decoder_dims = [32,64,128]
latent_dim = 20
decoder_activations = ['relu']*len(encoder_dims)
encoder_activations = ['relu']*len(decoder_dims)
vae = VAE(beta, encoder_dims, decoder_dims, latent_dim, decoder_activations,encoder_activations)
epochs = 25
batch_size = 128
save = True
name = 'DEEP2'
vae.train(epochs,batch_size,save,name)

#grande deluxe
print("-----------GRANDE DELUXE-------------")
encoder_dims = [512,256]
decoder_dims = [256,512]
latent_dim = 4
decoder_activations = ['relu']*len(encoder_dims)
encoder_activations = ['relu']*len(decoder_dims)
vae = VAE(beta, encoder_dims, decoder_dims, latent_dim, decoder_activations,encoder_activations)
epochs = 25
batch_size = 128
save = True
name = 'GRANDE_DELUXE'
vae.train(epochs,batch_size,save,name)

#grande deluxe2
print("-----------GRANDE DELUXE 2-------------")
encoder_dims = [512,256]
decoder_dims = [256,512]
latent_dim = 20
decoder_activations = ['relu']*len(encoder_dims)
encoder_activations = ['relu']*len(decoder_dims)
vae = VAE(beta, encoder_dims, decoder_dims, latent_dim, decoder_activations,encoder_activations)
epochs = 25
batch_size = 128
save = True
name = 'GRANDE_DELUXE2'
vae.train(epochs,batch_size,save,name)

#deep blue
print("-----------DEEP BLUE-------------")
encoder_dims = [256,128,128,64]
decoder_dims = [64,128,128,256]
latent_dim = 4
decoder_activations = ['relu']*len(encoder_dims)
encoder_activations = ['relu']*len(decoder_dims)
vae = VAE(beta, encoder_dims, decoder_dims, latent_dim, decoder_activations,encoder_activations)
epochs = 25
batch_size = 128
save = True
name = 'DEEP_BLUE'
vae.train(epochs,batch_size,save,name)

#deep blue2
print("-----------DEEP BLUE2-------------")
encoder_dims = [256,128,128,64]
decoder_dims = [64,128,128,256]
latent_dim = 20
decoder_activations = ['relu']*len(encoder_dims)
encoder_activations = ['relu']*len(decoder_dims)
vae = VAE(beta, encoder_dims, decoder_dims, latent_dim, decoder_activations,encoder_activations)
epochs = 250
batch_size = 128
save = False
name = 'DEEP_BLUE2'
vae.train(epochs,batch_size,save,name)

#megatron
print("-----------MEGATRON------------")
encoder_dims = [1024,512,256]
decoder_dims = [256,512,1024]
latent_dim = 20
decoder_activations = ['relu']*len(encoder_dims)
encoder_activations = ['relu']*len(decoder_dims)
vae = VAE(beta, encoder_dims, decoder_dims, latent_dim, decoder_activations,encoder_activations)
epochs = 25
batch_size = 128
save = True
name = 'MEGATRON'
vae.train(epochs,batch_size,save,name)

print("-----------GIGATRON------------")
encoder_dims = [1024,512,256,128]
decoder_dims = [128,256,512,1024]
latent_dim = 20
decoder_activations = ['relu']*len(encoder_dims)
encoder_activations = ['relu']*len(decoder_dims)
vae = VAE(beta, encoder_dims, decoder_dims, latent_dim, decoder_activations,encoder_activations)
epochs = 25
batch_size = 128
save = True
name = 'GIGATRON'
vae.train(epochs,batch_size,save,name)

#mr roboto
encoder_dims = [256,128]
decoder_dims = [128,256]
latent_dim = 20
decoder_activations = ['relu']*len(encoder_dims)
encoder_activations = ['relu']*len(decoder_dims)
vae = VAE(beta, encoder_dims, decoder_dims, latent_dim, decoder_activations,encoder_activations)
epochs = 200
batch_size = 64
save = True
name = 'Big_Papa'
cutoff = 25
vae.train(epochs,batch_size,save,name,cutoff)

cutoffs = [1,2,4,8,12]
min_dim = [64,128,256,512,1024]
bestloss = 100
bestnetwork = ''
bestmodel = None
latent_dims = [20]
for latent_dim in latent_dims:
    for dim in min_dim:
        encoder_dims = [dim,dim*2]
        decoder_dims = [dim*2,dim]
        #latent_dim = 20
        decoder_activations = ['relu']*len(encoder_dims)
        encoder_activations = ['relu']*len(decoder_dims)
        vae = VAE(beta, encoder_dims, decoder_dims, latent_dim, decoder_activations,encoder_activations)
        epochs = 50
        batch_size = 128
        save = False
        name = ''
        for cutoff in cutoffs:
            loss=vae.train(epochs,batch_size,save,name,cutoff)
            if loss<bestloss:
                bestloss = loss
                bestmodel = vae
                bestnetwork = str(dim) + ' x ' + str(dim//2) + ' CUTOFF: ' + str(cutoff) + ' LATENT DIM: ' + str(latent_dim)
print(bestnetwork)
print(bestloss)
#bestmodel.save_weights('best_model')