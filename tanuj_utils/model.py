from keras.models import Model
from keras.layers import BatchNormalization, Conv2D, Conv2DTranspose, MaxPooling2D, Dropout, UpSampling2D, Input, concatenate

def unet(
    input_shape,
    num_classes=1,
    use_batch_norm=True,
    upsample_mode='deconv'
    use_dropout_on_upsampling=False,
    dropout=0.3
    dropout_change_per_layer=0.0,
    filters=16,
    num_layers=4,
    output_activation='sigmoid'): # sigmoid or softmax
    
    x=Input(shape=input_shape)

    down_layers = []
    
    for l in range(num_layers):
        x = conv2d_block(inputs=x, fliters=filters, use_batch_norm=use_batch_norm, dropout=dropout)
        down_layers.append(x)
        x = MaxPooling2D(pool_size=(2,2)) (x)
        dropout += dropout_change_per_layer
        filters = filters*2 # double the number of filters with each layer
            
        
    x = conv2d_block(inputs=x, filters=filters, use_batch_norm=use_batch_norm, dropout=dropout)
    
    if not use_dropout_on_upsampling:
        dropout = 0.0
        dropout_change_per_layer = 0.0
        
    for conv in reversed(down_layers):
        filters //=2
        dropout -= dropout_change_per_layer
        x = Conv2DTranspose(filters,(2,2),strides=(2,2), padding='same') (x)
        x = concatenate([x,conv])
        x = conv2d_block(inputs=x, filters=filters, use_batch_norm=use_batch_norm, dropout=dropout)
        
    output = Conv2D(num_classes, (1,1), activation=output_activation) (x)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    return model