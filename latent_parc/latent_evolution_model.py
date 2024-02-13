
import numpy as np
import tensorflow as tf
import keras
from keras.layers import *
from keras import regularizers

def resnet_unit(feat_dim, kernel_size, x_in):
    # conv = Conv2D(feats, kernel, padding="same")
    res = keras.Sequential([
        Conv2D(feat_dim, kernel_size, padding = "same"
               ,kernel_initializer = 'he_uniform'
               ,bias_initializer = 'he_uniform',
               kernel_regularizer = regularizers.L1(l1=1e-4)
              ),
        ReLU(),
        Conv2D(feat_dim, kernel_size, padding = "same", 
               kernel_initializer = 'he_uniform',
               bias_initializer = 'he_uniform',
               kernel_regularizer = regularizers.L1(l1=1e-4))
    ])
    return ReLU()(x_in + res(x_in))

def resnet_block(feat_dim, reps, x_in):
    # Stage 2
    conv1 = Conv2D(feat_dim, 3, padding = "same", activation = 'relu', 
                   kernel_initializer = 'he_uniform', bias_initializer = 'he_uniform',
                   kernel_regularizer = regularizers.L1(l1=1e-4))(x_in)
    x = Conv2D(feat_dim, 3, padding = "same", activation = 'relu', 
                   kernel_initializer = 'he_uniform', bias_initializer = 'he_uniform',
                   kernel_regularizer = regularizers.L1(l1=1e-4))(conv1)
    for _ in range(reps):
        x = resnet_unit(feat_dim,3,x)
    x = MaxPooling2D(2,2)(x)
    x = BatchNormalization(axis=-1)(x)
    return x


def generator_unit(feats_in, feats_out, kernel, x):
    x = GaussianNoise(0.05)(x)
    x = Conv2D(feats_in, kernel, padding='same', activation= LeakyReLU(0.2),
                   kernel_initializer='he_uniform', bias_initializer='he_uniform',
                   kernel_regularizer=regularizers.L1(l1=1e-4))(x)
    x = resnet_unit(feats_in,3,x) 
        
   
    x = Conv2D(feats_out, kernel, padding='same', 
                      kernel_initializer='he_uniform', bias_initializer='he_uniform',
                      kernel_regularizer=regularizers.L1(l1=1e-4))(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2DTranspose(filters=feats_out, kernel_size=4, strides=2, padding='same')(x)
    x = BatchNormalization(axis=-1)(x)
    return x
    

def resnet_based_encoder(init_feat, input_shape):
    inputs = keras.Input(shape = input_shape)
    x = Conv2D(init_feat, 5, padding="same", activation = 'relu', 
               kernel_initializer='he_uniform', bias_initializer='he_uniform', 
               kernel_regularizer= regularizers.L1(l1=1e-4))(inputs)
    # x = MaxPooling2D(2,2)(x)
    # x = Dropout(0.01)(x)
    for i in range(3):
        if i < 2:
            x = resnet_block(init_feat*(2**i),1,x)
        elif i >= 2:
            x = resnet_block(32,1,x)
        #x = Dropout(0.3)(x)
    
    encoder = keras.Model(inputs, x)
    return encoder


def resnet_based_decoder(input_shape):
    latent_input = keras.Input(shape = input_shape)
    feat_ls = input_shape[2]
    latent_input = GaussianNoise(0.01)(latent_input)
    sp_bl_init = Conv2D(feat_ls, 3, padding = "same", activation = LeakyReLU(0.2),
                        kernel_initializer = 'he_uniform', bias_initializer = 'he_uniform',
                        kernel_regularizer = regularizers.L1(l1=1e-4))(latent_input)
    x = Dropout(0.01)(sp_bl_init)
    for i in range(3):
        x = generator_unit(feat_ls//(2**i), feat_ls//(2**(i+1)), 3, x)
        #x = Dropout(0.3)(x)

    recon = Conv2D(1, 3, padding = "same",
                    kernel_initializer = 'he_uniform', bias_initializer = 'he_uniform',
                    kernel_regularizer = regularizers.L1(l1=1e-4))(x)
    decoder = keras.Model(latent_input, recon)
    return decoder

# def latent_evolution_model(input_shape):
#     latent_input = keras.Input(shape = input_shape)
#     feat_ls = input_shape[2]
#     conv1 = Conv2D(feat_ls, 3, padding = 'same', activation = 'relu',
#                    kernel_initializer='he_uniform', bias_initializer='he_uniform', 
#                    kernel_regularizer=regularizers.L1(l1=1e-4))(latent_input)
#     conv2 = Conv2D(feat_ls, 3, padding = 'same', activation = 'relu',
#                    kernel_initializer = 'he_uniform', bias_initializer = 'he_uniform',
#                    kernel_regularizer = regularizers.L1(l1=1e-4))(conv1)
#     conv3 = Conv2D(feat_ls, 3, padding = 'same', activation = 'relu',
#                    kernel_initializer = 'he_uniform', bias_initializer = 'he_uniform', 
#                    kernel_regularizer = regularizers.L1(l1=1e-4))(conv2)
#     conv4 = Conv2D(feat_ls, 3, padding = 'same', activation = 'relu',
#                     kernel_initializer = 'he_uniform', bias_initializer = 'he_uniform', 
#                     kernel_regularizer = regularizers.L1(l1=1e-4))(conv3)
#     output = latent_input + conv4
#     latent_out = keras.Model(latent_input, output)
#     return latent_out



def latent_evolution_model(input_shape):
    latent_input = keras.Input(shape = input_shape)
    feat_ls = input_shape[2]
    x = Conv2D(feat_ls, 3, padding = 'same', activation = 'relu',
                   kernel_initializer='he_uniform', bias_initializer='he_uniform', 
                   kernel_regularizer=regularizers.L1(l1=1e-4))(latent_input)
    x = Conv2D(128, 3, padding = 'same', activation = 'relu',
                   kernel_initializer = 'he_uniform', bias_initializer = 'he_uniform',
                   kernel_regularizer = regularizers.L1(l1=1e-4))(x)
    x = Conv2D(256, 3, padding = 'same', activation = 'relu',
                   kernel_initializer = 'he_uniform', bias_initializer = 'he_uniform',
                   kernel_regularizer = regularizers.L1(l1=1e-4))(x)
    x = Conv2D(256, 3, padding = 'same', activation = 'relu',
                   kernel_initializer = 'he_uniform', bias_initializer = 'he_uniform',
                   kernel_regularizer = regularizers.L1(l1=1e-4))(x)
    x = Conv2D(128, 3, padding = 'same', activation = 'relu',
                   kernel_initializer = 'he_uniform', bias_initializer = 'he_uniform',
                   kernel_regularizer = regularizers.L1(l1=1e-4))(x)
    x = Conv2D(feat_ls, 3, padding = 'same', activation = 'relu',
                   kernel_initializer = 'he_uniform', bias_initializer = 'he_uniform', 
                   kernel_regularizer = regularizers.L1(l1=1e-4))(x)
    output = latent_input + x
    latent_out = keras.Model(latent_input, output)
    return latent_out

# Isometric Regularization Loss
def relaxed_distorsion_measure(decoder, z):
    bs = tf.shape(z)[0] 
    v = tf.random.normal(tf.shape(z))
        
    with tf.autodiff.ForwardAccumulator(primals=z, tangents=v) as acc:
        ae_recon = decoder(z)
    Jv = acc.jvp(ae_recon)
    TrG = tf.reduce_mean(tf.reduce_sum(tf.reshape(Jv, [bs, -1])**2, axis=1))
        
    with tf.GradientTape() as tape:
        tape.watch(z)
        ae_recon_1 = decoder(z)
    JTJv = tape.gradient(ae_recon_1, z, output_gradients = Jv)
    TrG2 = tf.reduce_mean(tf.reduce_sum(tf.reshape(JTJv, [bs, -1])**2, axis=1))
        
    return TrG2/(TrG**2)

def time_derivative(state_in, state_out):
    return state_out - state_in

def gradient_distortion(source_point, source_tangent, mapping, target_tangent):
    with tf.autodiff.ForwardAccumulator(primals=source_point, tangents=source_tangent) as acc:
        target = mapping(source_point)
    mapped_tangent = acc.jvp(target)
    loss = tf.keras.losses.MeanSquaredError()(mapped_tangent, target_tangent)
    return loss


### Shahab's modified version of Phong's PILE

class PILE_v2(keras.Model):
    def __init__(self, field, init_feat, input_shape, loss_weights, **kwargs):
        super(PILE_v2, self).__init__(**kwargs)
        self.field = field
        self.init_feat = init_feat
        self.in_shape = input_shape
        self.loss_weights = loss_weights
        ###
        enc = resnet_based_encoder(self.init_feat, (self.in_shape[0], self.in_shape[1], 1))
        setattr(self, 'encoder_'+self.field, enc)
        ###
        test_input = tf.random.uniform((5, self.in_shape[0], self.in_shape[1], 1))
        self.latent_shape = (enc(test_input)).numpy().shape[1:]
        dec = resnet_based_decoder(self.latent_shape)
        setattr(self, 'decoder_'+self.field, dec)    
        ###
        self.latent_evolution = latent_evolution_model((self.latent_shape[0], self.latent_shape[1], self.latent_shape[2]))
        # loss define
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.in_ae_loss_tracker = keras.metrics.Mean(name="in_ae_loss")
        self.out_ae_loss_tracker = keras.metrics.Mean(name="out_ae_loss")
        self.latent_loss_tracker = keras.metrics.Mean(name="latent_loss")
        self.recon_loss_tracker = keras.metrics.Mean(name="recon_loss")


    @property
    def metrics(self):
        return [
        self.total_loss_tracker,
        self.in_ae_loss_tracker,
        self.out_ae_loss_tracker,
        self.latent_loss_tracker,
        self.recon_loss_tracker
        ]
    
    

    def train_step(self, data):

        input_state = data[0]
        output_state = data[1]
        ###
        total_loss = 0
        in_ae_loss = 0
        out_ae_loss = 0
        latent_loss = 0
        recon_loss = 0
        criterion = tf.keras.losses.MeanSquaredError()
        ###
        with tf.GradientTape() as tape:
            # AE loss
            latent_field_in = getattr(self, 'encoder_'+self.field)(input_state)
            ae_recon_field_in = getattr(self, 'decoder_'+self.field)(latent_field_in)
            in_ae_loss += criterion(input_state, ae_recon_field_in)
            ###
            latent_field_out = getattr(self, 'encoder_'+self.field)(output_state)
            ae_recon_field_out = getattr(self, 'decoder_'+self.field)(latent_field_out)
            out_ae_loss += criterion(output_state, ae_recon_field_out)
            ###
            ae_loss = in_ae_loss + out_ae_loss
            
            # Latent Evolution loss
            pred_latent_field_out = self.latent_evolution(latent_field_in)
            latent_loss = criterion(pred_latent_field_out, latent_field_out)

            # Dynamics Reconstruction loss
            recon_evolved_field = getattr(self, 'decoder_'+self.field)(pred_latent_field_out)
            recon_loss += criterion(output_state, recon_evolved_field)

            #           
            total_loss =  (self.loss_weights[0] * ae_loss) + (self.loss_weights[1] * latent_loss) + (self.loss_weights[2] * recon_loss) 
            
            # # IR loss
            # in_ir_loss = relaxed_distorsion_measure(self.decoder, latent_state_in)
            # out_ir_loss = relaxed_distorsion_measure(self.decoder, latent_state_out)
            # ir_loss = in_ir_loss + out_ir_loss

            # # Gradient distortion loss
            # dec_gd_loss = gradient_distortion(latent_state_in,
            #                                   time_derivative(latent_state_in, latent_state_out),
            #                                   self.decoder,
            #                                   time_derivative(input_state, output_state))
            
            # enc_gd_loss = gradient_distortion(input_state,
            #                                   time_derivative(input_state, output_state),
            #                                   self.encoder,
            #                                   time_derivative(latent_state_in, latent_state_out))
            
            # gd_loss = dec_gd_loss + enc_gd_loss

            
        self.total_loss_tracker.update_state(total_loss)
        self.in_ae_loss_tracker.update_state(in_ae_loss)
        self.out_ae_loss_tracker.update_state(out_ae_loss)
        self.latent_loss_tracker.update_state(latent_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))


        return {
            "total_loss": self.total_loss_tracker.result(),
            "in_ae_loss": self.in_ae_loss_tracker.result(),
            "out_ae_loss": self.out_ae_loss_tracker.result(),
            "latent_loss": self.latent_loss_tracker.result(),
            "recon_loss": self.recon_loss_tracker.result()
        }
    
    def predict(self, field_gt, batch_size=1, **kwargs):
        # AE-reconstruction of one of the fields
        enc = getattr(self, 'encoder_'+self.field)
        dec = getattr(self, 'decoder_'+self.field)
        field_ae_recon = (dec(enc(field_gt))).numpy()        
        field_concat = np.squeeze(np.stack([field_gt, field_ae_recon], axis=0)) 
        ###
        return field_concat

###
def state_norm(a):
    # "a" has the shape (samples, im_height, im_width, channels)
    b = tf.norm(a, ord=2, axis=3)
    b = tf.reduce_sum(b, axis=[1,2])/(a.shape[1]*a.shape[2])
    for i in range(1,4):
        b = tf.expand_dims(b, axis=i)
    return b    

    
####
class PILE_v3(PILE_v2):
    def __init__(self, ts, field, init_feat, input_shape, loss_weights, **kwargs):
        super(PILE_v3, self).__init__(field, init_feat, input_shape, loss_weights, **kwargs)
        self.ts = ts
        ##
        self.total_loss_tracker_v3 = keras.metrics.Mean(name="total_loss_v3")
        self.ae_loss_tracker_v3 = keras.metrics.Mean(name="ae_loss_v3")
        self.latent_loss_tracker_v3 = keras.metrics.Mean(name="latent_loss_v3")
        self.recon_loss_tracker_v3 = keras.metrics.Mean(name="recon_loss_v3")
        # self.iso_loss_tracker_v3 = keras.metrics.Mean(name="iso_loss_v3")
        
    @property
    def metrics(self):
        return [
        self.total_loss_tracker_v3,
        self.ae_loss_tracker_v3,
        self.latent_loss_tracker_v3,
        self.recon_loss_tracker_v3
        # self.iso_loss_tracker_v3
        ]


    def train_step(self, data):

        init_field = data[0]
        field_traj = data[1]
        ###
        total_loss = 0
        ae_loss = 0
        latent_loss = 0
        recon_loss = 0
        # iso_loss = 0
        criterion = tf.keras.losses.MeanSquaredError()
        ###
        with tf.GradientTape() as tape:
            enc = getattr(self, 'encoder_'+self.field)
            dec = getattr(self, 'decoder_'+self.field)
            current_latent_field = enc(init_field)
            ae_loss = criterion(init_field, dec(current_latent_field))
            for i in range(self.ts):
                ###
                iter_ae_loss = criterion(field_traj[:,:,:,:,i], dec(enc(field_traj[:,:,:,:,i])))
                ###
                next_latent_field_evol = self.latent_evolution(current_latent_field)
                next_latent_field_enc = enc(field_traj[:,:,:,:,i])
                norm = state_norm(next_latent_field_enc)
                iter_latent_loss = criterion(tf.divide(next_latent_field_evol, norm), tf.divide(next_latent_field_enc, norm))
                ###
                next_field_recon = dec(next_latent_field_evol)
                iter_recon_loss = criterion(field_traj[:,:,:,:,i], next_field_recon)
                ###
                # iter_iso_loss = relaxed_distorsion_measure(dec, current_latent_field)
                ###
                ae_loss += iter_ae_loss
                latent_loss += iter_latent_loss
                recon_loss += iter_recon_loss
                # iso_loss += iter_iso_loss
                ###
                current_latent_field = next_latent_field_evol

            #           
            total_loss =  (self.loss_weights[0] * ae_loss) + (self.loss_weights[1] * latent_loss) + (self.loss_weights[2] * recon_loss) 
#                         + (self.loss_weights[3] * iso_loss)
            

            
        self.total_loss_tracker_v3.update_state(total_loss)
        self.ae_loss_tracker_v3.update_state(ae_loss)
        self.latent_loss_tracker_v3.update_state(latent_loss)
        self.recon_loss_tracker_v3.update_state(recon_loss)
        # self.iso_loss_tracker_v3.update_state(iso_loss)
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))


        return {
            "total_loss": self.total_loss_tracker_v3.result(),
            "ae_loss": self.ae_loss_tracker_v3.result(),
            "latent_loss": self.latent_loss_tracker_v3.result(),
            "recon_loss": self.recon_loss_tracker_v3.result()
            # "iso_loss": self.iso_loss_tracker_v3.result()
        }
    
    def predict_v3(self, ts, field_gt):
        enc = getattr(self, 'encoder_'+self.field)
        dec = getattr(self, 'decoder_'+self.field)
        evol = self.latent_evolution
        init_latent_field = enc(field_gt)
        next_latent_field = evol(init_latent_field)
        for _ in range(ts-1):
            next_latent_field = evol(next_latent_field)
        recon_field = dec(next_latent_field)
        field_concat = tf.stack([field_gt[ts:,:,:,:], recon_field[:-ts,:,:,:]], axis=0)
        return field_concat.numpy().squeeze()

    
#### Phong's Inception module for learning the latent dynamics

def latent_physics_model(input_shape):
    #z_m = keras.Input(shape = (14,28,64))
    z_t = keras.Input(shape = input_shape)
    
#     concat = Concatenate()([z_m,z_t])
    #z_m_noise = GaussianNoise(0.05)(z_m)  
    z_t_noise = GaussianNoise(0.05)(z_t)  

    # spade block
    sp_bl = Dropout(0.05)(z_t_noise)
    

    # inception block 1
    conv11 = Conv2D(128,1, activation = 'relu', padding = 'same',
                    kernel_initializer = 'he_uniform', bias_initializer = 'he_uniform')(sp_bl)
    
    conv21 = Conv2D(128,1, activation = 'relu', padding = 'same',
                    kernel_initializer = 'he_uniform', bias_initializer = 'he_uniform')(sp_bl)
    conv21 = Conv2D(128,3, activation = 'relu', padding = 'same',
                    kernel_initializer = 'he_uniform', bias_initializer = 'he_uniform')(conv21)
    
    conv31 = Conv2D(128,1, activation = 'relu', padding = 'same', 
                    kernel_initializer = 'he_uniform', bias_initializer = 'he_uniform')(sp_bl)
    conv31 = Conv2D(128,5, activation = 'relu', padding = 'same',
                    kernel_initializer = 'he_uniform', bias_initializer = 'he_uniform')(conv31)
    
    incept1 = Concatenate()([Concatenate()([conv11,conv21]),conv31])
    incept1 = Dropout(0.05)(incept1)

    # Nonlinear mapping
    conv_out1 = Conv2D(256,1, activation = 'relu', padding = 'same',
                       kernel_initializer = 'he_uniform', bias_initializer = 'he_uniform')(incept1)
    conv_out1 = Dropout(0.05)(conv_out1)
    
    conv_out2 = Conv2D(512,1, activation = 'relu', padding = 'same',
                       kernel_initializer = 'he_uniform', bias_initializer = 'he_uniform')(conv_out1)    
    conv_out2 = Dropout(0.05)(conv_out2)

    conv_out3 = Conv2D(1024,1, activation = 'relu', padding = 'same',
                       kernel_initializer = 'he_uniform', bias_initializer = 'he_uniform')(conv_out2)
    conv_out3 = Dropout(0.05)(conv_out3)

    conv_out = Conv2D(input_shape[2], 1, padding = 'same', 
                      kernel_initializer = 'he_uniform', bias_initializer = 'he_uniform')(conv_out3)

    # out = z_t + conv_out
  
    latent_out = keras.Model(z_t, conv_out)
    return latent_out



### Shahab's modified version of Phong's LatentPARC in which we have three independant autoencoders
### for Temperature, Pressure and Microstructure fields

class LatentPARC_v2(keras.Model):
    def __init__(self, ts, init_feat, input_shape, loss_weights, **kwargs):
        super(LatentPARC_v2, self).__init__(**kwargs)
        self.init_feat = init_feat
        self.in_shape = input_shape
        self.ts = ts
        self.loss_weights = loss_weights
        ###
        self.fields = ['temperature', 'pressure', 'microstructure']
        encoder = resnet_based_encoder(self.init_feat, (self.in_shape[0], self.in_shape[1], 1))
        for field in self.fields:
            setattr(self, 'encoder_'+field, encoder)
            enc = getattr(self, 'encoder_'+field)
            enc.load_weights('../NN_weights/Latent_PARC/encoder_'+field+'.h5')
            enc.trainable = False
        ###
        test_input = tf.random.uniform((5, self.in_shape[0], self.in_shape[1], 1))
        self.latent_shape = (enc(test_input)).numpy().shape[1:]
        decoder = resnet_based_decoder(self.latent_shape)
        for field in self.fields:
            setattr(self, 'decoder_'+field, decoder)
            dec = getattr(self, 'decoder_'+field)
            dec.load_weights('../NN_weights/Latent_PARC/decoder_'+field+'.h5')
            dec.trainable = False 
        ###
        # self.latent_physics = latent_physics_model((self.latent_shape[0], self.latent_shape[1], self.latent_shape[2]*3))
        self.latent_physics = latent_evolution_model((self.latent_shape[0], self.latent_shape[1], self.latent_shape[2]*3))
        # loss define
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.latent_loss_tracker = keras.metrics.Mean(name="latent_loss")
        self.recon_loss_tracker = keras.metrics.Mean(name="recon_loss")


    @property
    def metrics(self):
        return [
        self.total_loss_tracker,
        self.latent_loss_tracker,
        self.recon_loss_tracker
        ]
    
    

    def train_step(self, data):

        init_latent_state = data[0][0]
        latent_state_traj = data[1][0]
        state_traj_gt = data[1][1]
        ###
        total_loss = 0
        latent_loss = 0
        recon_loss = 0
        criterion = tf.keras.losses.MeanSquaredError()
        ###
        with tf.GradientTape() as tape:
            current_latent_state = init_latent_state
            for i in range(self.ts):
                next_latent_state = self.latent_physics(current_latent_state)
                norm = state_norm(latent_state_traj[:,:,:,:,i])
                iter_latent_loss = criterion(tf.divide(next_latent_state, norm), tf.divide(latent_state_traj[:,:,:,:,i], norm))
                latent_loss += iter_latent_loss
                ##
                for j, field in enumerate(self.fields):
                    recon_evolved_field = getattr(self, 'decoder_'+field)(next_latent_state[:,:,:,j*self.latent_shape[2]:(j+1)*self.latent_shape[2]])
                    iter_recon_loss = criterion(recon_evolved_field, state_traj_gt[:,:,:,j:j+1,i])
                    recon_loss += iter_recon_loss
                ##
                current_latent_state = next_latent_state

            total_loss = (self.loss_weights[0] * latent_loss) + (self.loss_weights[1] * recon_loss) 
        ###    
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))    

        self.total_loss_tracker.update_state(total_loss)
        self.latent_loss_tracker.update_state(latent_loss)
        self.recon_loss_tracker.update_state(recon_loss)        
        
        return {
            "total_loss": self.total_loss_tracker.result(),
            "latent_loss": self.latent_loss_tracker.result(),
            "recon_loss": self.recon_loss_tracker.result()
        }
    
    
    def predict(self, input_state_gt, evol_step, batch_size=1, **kwargs):
        ###
        input_latent_state = []
        for i, f in enumerate(self.fields):
            input_latent_field = getattr(self, 'encoder_'+f)(input_state_gt[:,:,:,i:i+1])
            input_latent_state.append(input_latent_field)
        input_latent_state = tf.concat(input_latent_state, 3)
        ###
        # The latent state trajectory resulting from application of the latent evolution "evol_step" time-steps to the input ground-truth state
        # latent_state_traj = (samples, im_height, im_width, latent_state, micro-time-steps)
        latent_state_traj = [input_latent_state]
        next_latent_state = input_latent_state
        for _ in range(evol_step):
            next_latent_state = self.latent_physics(next_latent_state)
            latent_state_traj.append(next_latent_state)
        latent_state_traj = tf.stack(latent_state_traj, axis=4)
        ###
        state_evol_traj = []
        for t in range(evol_step+1):
            state_evol = []
            for i, f in enumerate(self.fields):
                field_evol = getattr(self, 'decoder_'+f)(latent_state_traj[:,:,:, i*self.latent_shape[2]:(i+1)*self.latent_shape[2],t])
                state_evol.append(field_evol)
            state_evol = tf.concat(state_evol, axis=3)
            state_evol_traj.append(state_evol)
        state_evol_traj = tf.stack(state_evol_traj, axis=4)
        ###
        return state_evol_traj.numpy(), latent_state_traj.numpy()