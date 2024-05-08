import numpy as np

class BaseData:
    def __init__(self, **kwargs):
        super(BaseData, self).__init__(**kwargs)

    # Normalization
    def data_normalization(self, input_data,no_of_channel):
        norm_data = np.zeros(input_data.shape)
        min_val = []
        max_val = []
        for i in range(no_of_channel):
            norm_data[:,:,:,i::no_of_channel] = ((input_data[:,:,:,i::no_of_channel] - np.amin(input_data[:,:,:,i::no_of_channel])) / (np.amax(input_data[:,:,:,i::no_of_channel]) - np.amin(input_data[:,:,:,i::no_of_channel])) + 1E-9)
            min_val.append(np.amin(input_data[:,:,:,i::no_of_channel]))
            max_val.append(np.amax(input_data[:,:,:,i::no_of_channel]))
        return norm_data, min_val, max_val

    def data_normalization_with_value(self, input_data, min_val, max_val, no_of_channel):
        norm_data = np.zeros(input_data.shape)
        for i in range(no_of_channel):
            norm_data[:,:,:,i::no_of_channel] = ((input_data[:,:,:,i::no_of_channel] - min_val[i]) / (max_val[i] - min_val[i] + 1E-9))
        return norm_data

    def data_denormalization(self, input_data, min_val, max_val, no_of_channel):
        denorm_data = np.zeros(input_data.shape)
        for i in range(no_of_channel):
            denorm_data[:,:,:,i::no_of_channel] = (input_data[:,:,:,i::no_of_channel] * (max_val[i] - min_val[i] + 1E-9)) + min_val[i]
        return denorm_data