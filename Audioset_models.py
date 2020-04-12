
def model_import(ind, n_filters, input_shape, n_seed=15000321):
    import numpy as np
    import keras.backend as K
    np.random.seed(n_seed)
    K.tf.set_random_seed(n_seed)
    if ind == 'Gen_LSTM':
        x_in = Input(shape= input_shape);
        x,state_h, state_c = CuDNNLSTM(n_filters, return_sequences=True, return_state = True ,
                 activity_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01),
                 name='encoder')(x_in);
        x = CuDNNLSTM(n_mel,return_sequences=True)(x, initial_state=[state_h, state_c])
    
    if ind == 'Gen_CNN':
        # https://arxiv.org/pdf/1603.05027.pdf
        from keras.layers import Input, Conv1D, BatchNormalization, Dropout, Activation, MaxPooling1D, Add, Flatten, Dense
        from keras.models import Model        
        
        x_in = Input(shape= input_shape);
        x = Conv1D(n_filters, kernel_size = 64, padding='same')(x_in)
        x = BatchNormalization()(x)
        x = Dropout(rate = .5)(x)
        x = Activation('relu')(x)
        x = Conv1D(n_filters, kernel_size = 1, padding='same')(x)
        x = BatchNormalization()(x)
        x = Add()([x, x_in])    
        x = Dropout(rate = .5)(x)
        x = Activation('relu')(x)
        
        x_pool = MaxPooling1D()(x)        
        
        x = Conv1D(n_filters, kernel_size = 32, padding='same')(x_pool)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(rate = .5)(x)
        x = Conv1D(n_filters, kernel_size = 1, padding='same')(x)
        x = BatchNormalization()(x)
        x = Add()([x, x_pool])   
        x = Activation('relu')(x)
        x = Dropout(rate = .5)(x)
    
        x_pool = MaxPooling1D()(x)
        
        
        x = Flatten()(x_pool)
        x = Dense(64)(x)
        x = BatchNormalization()(x)
        x = Dropout(rate = .5)(x)
        x = Activation('relu')(x)
        
        x = Dense(2)(x)
        x = BatchNormalization()(x)
        x = Dropout(rate = .5)(x)
        x = Activation('sigmoid')(x)
        
    model = Model(inputs=x_in, outputs=x)
#     model.summary()
    return model