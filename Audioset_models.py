
def model_import(ind, n_filters, input_shape,output_shape = 2, n_seed=15000321):
    import numpy as np
    import keras.backend as K
    np.random.seed(n_seed)
    K.tf.set_random_seed(n_seed)
    from keras.models import Model        
    
    if ind == 'Gen_LSTM':
        x_in = Input(shape= input_shape);
        x,state_h, state_c = CuDNNLSTM(n_filters, return_sequences=True, return_state = True ,
                 activity_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01),
                 name='encoder')(x_in);
        x = CuDNNLSTM(n_mel,return_sequences=True)(x, initial_state=[state_h, state_c])
    
    if ind == 'Gen_CNN':
        # https://arxiv.org/pdf/1603.05027.pdf
        from keras.layers import Input, Conv1D, BatchNormalization, Dropout, Activation, MaxPooling1D, Add, Flatten, Dense
        
        
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
        
        x = Dense(output_shape)(x)
        x = BatchNormalization()(x)
        x = Dropout(rate = .5)(x)
        x = Activation('sigmoid')(x)
        
    if ind == 'Gen_CNN_small':
        # https://arxiv.org/pdf/1603.05027.pdf
        from keras.layers import Input, Conv1D, BatchNormalization, Dropout, Activation, MaxPooling1D, Add, Flatten, Dense
        
        
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
               
        
        x = Flatten()(x_pool)
        
        x = Dense(output_shape)(x)
        x = BatchNormalization()(x)
        x = Dropout(rate = .5)(x)
        x = Activation('sigmoid')(x)
        
    if ind == 'Gen_Attention':
        from keras.layers import Input, Conv1D, BatchNormalization, Dropout, Activation, MaxPooling1D, Add, Flatten, Dense
        from keras.layers import CuDNNGRU, Input, Dense, BatchNormalization, Activation, TimeDistributed
        import keras.backend as K
        import numpy as np
        from keras_attention.models.custom_recurrents import AttentionDecoder
        from keras.layers.wrappers import Bidirectional
        from keras.models import load_model
        
        x_in = Input(shape = (input_shape[0],input_shape[1]), name='in_signal');             
        x = Bidirectional(CuDNNGRU(n_filters, return_sequences=True, name='encoder'), trainable=True)(x_in)
        #x = BatchNormalization()(x)
        x_attention = AttentionDecoder(int(n_filters/2), input_shape[1],name='out_main', trainable=True)(x)
        
        #model_CNN = load_model('logs/Gen_CNN_MULTI-2020-05-05_13.27.04.380845/Gen_CNN.h5')
        model_CNN = load_model('logs/Gen_CNN_small_MULTI-2020-05-06_19.23.30.429287/Gen_CNN_small.h5')
        # We freeze the weights of the CNN model.
#         for i in model_CNN.layers:
#             i.trainable = False;
        #x = model_CNN()(x_attention)
        x = model_CNN(inputs = x_attention)
        
        
    if ind == 'Gen_Attention_BIG':
        from keras.layers import Input, Conv1D, BatchNormalization, Dropout, Activation, MaxPooling1D, Add, Flatten, Dense
        from keras.layers import CuDNNGRU, Input, Dense, BatchNormalization, Activation, TimeDistributed, CuDNNLSTM
        import keras.backend as K
        import numpy as np
        from keras_attention.models.custom_recurrents import AttentionDecoder
        from keras.layers.wrappers import Bidirectional
        from keras.models import load_model
        
        x_in = Input(shape = (input_shape[0],input_shape[1]), name='in_signal');             
        x_attention = CuDNNLSTM(n_filters, return_sequences=True, name='encoder')(x_in)
        #x = BatchNormalization()(x)
        #x_attention = AttentionDecoder(int(n_filters/2), input_shape[1],name='out_main', trainable=True)(x)
        
        model_CNN = load_model('logs/Gen_CNN_MULTI-2020-05-05_13.27.04.380845/Gen_CNN.h5')
        x = model_CNN(inputs = x_attention)
        
    if ind == 'Gen_Attention_raw':
        from keras.layers import Input, Conv1D, BatchNormalization, Dropout, Activation, MaxPooling1D, Add, Flatten, Dense
        from keras.layers import CuDNNGRU, Input, Dense, BatchNormalization, Activation, TimeDistributed, CuDNNLSTM, Multiply
        import keras.backend as K
        import numpy as np
        from keras_attention.models.custom_recurrents import AttentionDecoder
        from keras.layers.wrappers import Bidirectional
        from keras.models import load_model
        from keras.regularizers import l1
        
        x_in = Input(shape = (input_shape[0],input_shape[1]), name='in_signal');             
        x_attention = Bidirectional(CuDNNLSTM(int(n_filters/2), return_sequences=True, name='encoder', activity_regularizer=l1(.1)))(x_in)
        x_attention = TimeDistributed(Dense(n_filters,activation='sigmoid'))(x_attention)
        #x = BatchNormalization()(x)
        x_out = Multiply()([x_attention, x_in])
        
        model_CNN = load_model('logs/Gen_CNN_MULTI-2020-05-05_13.27.04.380845/Gen_CNN.h5')
        x = model_CNN(inputs = x_out)
    
    if ind == 'Pre_feature_singleLSTM_BIG':
        from keras.layers import Input, Conv1D, BatchNormalization, Dropout, Activation, MaxPooling1D, Add, Flatten, Dense
        from keras.layers import CuDNNGRU, Input, Dense, BatchNormalization, Activation, TimeDistributed, CuDNNLSTM, Multiply
        import keras.backend as K
        import numpy as np
        from keras_attention.models.custom_recurrents import AttentionDecoder
        from keras.layers.wrappers import Bidirectional
        from keras.models import load_model
        from keras.regularizers import l1
        
        x_in = Input(shape = (input_shape[0],input_shape[1]), name='in_signal');          
        
        x = Conv1D(n_filters, kernel_size = 64, padding='same')(x_in)
        x = BatchNormalization()(x)
        x = Dropout(rate = .5)(x)
        x = Activation('relu', name='pre')(x)
        
        x = CuDNNLSTM(n_filters, return_sequences=True, name='encoder', activity_regularizer=l1(.1))(x)        
        x_out = BatchNormalization()(x)
        
        model_CNN = load_model('logs/Gen_CNN_MULTI-2020-05-05_13.27.04.380845/Gen_CNN.h5')
        x = model_CNN(inputs = x_out)
    
        
        
    model = Model(inputs=x_in, outputs=x)
#     model.summary()
    return model