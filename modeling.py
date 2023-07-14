import math
import os
import struct
import models
import sys
import joblib
import numpy as np
import pandas as pd
from copy import deepcopy
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Dense
from keras.models import Sequential
from keras.models import load_model
from keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import csv
from tensorflow.keras import backend as K
from sklearn.preprocessing import RobustScaler
from keras.activations import elu
from tensorflow.keras.layers import PReLU
from keras.activations import selu
import tensorflow as tf
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import QuantileTransformer
from keras.layers import BatchNormalization
from sklearn.preprocessing import PowerTransformer


global epoch_count


delimiter = b'\x48\x48'
delimiter_size = len(delimiter)

relearn = True
load = True
epoch_count = 1500
save_iterations = 20
min_lr = 1e-22

#Neural Network parameters
batch = 12345
lr = 0.01
early_stop_patience = epoch_count
reduce_lr_patience = 8
reduce_lr_factor = 0.2



def create_model_complex():
    model = Sequential()
    model.add(Dense(16384, input_dim=3, activation=PReLU()))
    model.add(BatchNormalization())
    model.add(Dense(16384, activation=PReLU()))
    model.add(BatchNormalization())
    model.add(Dense(8192, activation=PReLU()))
    model.add(BatchNormalization())
    model.add(Dense(8192, activation=PReLU()))
    model.add(BatchNormalization())
    model.add(Dense(8192, activation=PReLU()))
    model.add(BatchNormalization())
    model.add(Dense(8192, activation=PReLU()))
    model.add(BatchNormalization())
    model.add(Dense(8192, activation=PReLU()))
    model.add(BatchNormalization())
    model.add(Dense(8192, activation=PReLU()))
    model.add(BatchNormalization())
    model.add(Dense(8192, activation=PReLU()))
    model.add(BatchNormalization())
    model.add(Dense(8192, activation=PReLU()))
    model.add(BatchNormalization())
    model.add(Dense(4096, activation=PReLU()))
    model.add(BatchNormalization())
    model.add(Dense(4096, activation=PReLU()))
    model.add(BatchNormalization())
    model.add(Dense(4096, activation=PReLU()))
    model.add(BatchNormalization())
    model.add(Dense(4096, activation=PReLU()))
    model.add(BatchNormalization())
    model.add(Dense(4096, activation=PReLU()))
    model.add(BatchNormalization())
    model.add(Dense(2048, activation=PReLU()))
    model.add(BatchNormalization())
    model.add(Dense(2048, activation=PReLU()))
    model.add(BatchNormalization())
    model.add(Dense(2048, activation=PReLU()))
    model.add(BatchNormalization())
    model.add(Dense(2048, activation=PReLU()))
    model.add(BatchNormalization())
    model.add(Dense(1024, activation=PReLU()))
    model.add(BatchNormalization())
    model.add(Dense(1024, activation=PReLU()))
    model.add(BatchNormalization())
    model.add(Dense(1024, activation=PReLU()))
    model.add(BatchNormalization())
    model.add(Dense(512, activation=PReLU()))
    model.add(BatchNormalization())
    model.add(Dense(256, activation=PReLU()))
    model.add(BatchNormalization())
    model.add(Dense(128, activation=PReLU()))
    model.add(BatchNormalization())
    model.add(Dense(64, activation=PReLU()))
    model.add(BatchNormalization())
    model.add(Dense(32, activation=PReLU()))
    model.add(BatchNormalization())
    model.add(Dense(1))

    optimizer = Adam(learning_rate=lr)

    model.compile(loss='mean_absolute_error', optimizer=optimizer)
    return model

def extract_data_from_block(block):
    raw_data = block[:4]  # float data
    cell_format = '<f'
    parsed_float = struct.unpack(cell_format, raw_data)[0]  # unpack float
    if parsed_float is None or math.isnan(parsed_float):
        hex_string = raw_data.hex()
        print(f"Found NaN value; Hex representation: {hex_string}. Replacing with 0.")
        parsed_float = 0.0
    num_ints = len(block[4:]) // 4
    cell_format = '<' + 'i' * num_ints  # unpack the rest as signed integers
    raw_data = block[4:num_ints * 4 + 4]
    parsed_data = struct.unpack(cell_format, raw_data)
    row_data = [parsed_float]  # start with parsed float
    for i in range(num_ints):
        cell = parsed_data[i]
        if cell is None or math.isnan(cell):
            hex_string = raw_data[i*4:(i+1)*4].hex()
            print(f"Found NaN value; Hex representation: {hex_string}. Replacing with 0.")
            cell = 0
        row_data.append(cell)
    return row_data

def find_delimiter_indices(byte_data, delimiter=b'\x48\x48'):
    delimiter_size = len(delimiter)
    delimiter_indices = []
    position = 0
    while position < len(byte_data):
        delimiter_position = byte_data.find(delimiter, position)
        if delimiter_position == -1:
            break
        delimiter_indices.append(delimiter_position)
        position = delimiter_position + 92
    return delimiter_indices

def parse_byte_data_into_rows(byte_data, delimiter_indices):
    rows = []
    row_size = 92
    for i in range(len(delimiter_indices)-1):
        start_index = delimiter_indices[i] + delimiter_size
        end_index = start_index + row_size
        try:
            row = extract_data_from_block(byte_data[start_index:end_index])
            rows.append(row)
        except Exception as e:
            print(f"Error parsing block into row: {e}")
            continue
    last_start_index = delimiter_indices[-1] + delimiter_size
    try:
        last_row = extract_data_from_block(byte_data[last_start_index:])
        rows.append(last_row)
    except Exception as e:
        print(f"Error parsing block into row: {e}")
    return rows


import logging

logging.basicConfig(filename='error.log', level=logging.ERROR, format='%(asctime)s %(levelname)s: %(message)s')


def main():
    input_file_path = 'scale'
    csv_file_path = 'scale_original.csv'
    
    try:
        with open(input_file_path, 'rb') as input_file:
            byte_data = input_file.read()
    except Exception as e:
        print(f"Error reading input file: {e}")
        return

    delimiter_indices = find_delimiter_indices(byte_data)
    if not delimiter_indices:
        print("No delimiter indices found.")
        return

    rows = parse_byte_data_into_rows(byte_data, delimiter_indices)
    if not rows:
        print("No rows were parsed.")
        return

    try:
        original_values = pd.read_csv(csv_file_path, header=0, usecols=range(1,23)).values
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    # Calculate actual time steps and wavelengths:
    time_steps = np.linspace(0, 5, original_values.shape[0])
    wavelengths = np.linspace(190, 400, original_values.shape[1])

    # Load the best losses, if exists
    try:
        best_losses = joblib.load("best_losses.joblib")
    except:
        best_losses = [np.inf] * 23  # initialize with infinity
        joblib.dump(best_losses, "best_losses.joblib")

    best_models = [None]*23

    training_models = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]

    try:
        time_steps = np.linspace(0, 5, len(rows))
        relative_loss = np.inf
        for position in training_models:
            global epoch_count
            best_neural_loss = best_losses[position]
            parsed_values = [row[position] for row in rows if len(row) == 23]
            parsed_values_reshaped = np.array(parsed_values).reshape(-1, 1)

            original_values_column = original_values[:, position-1].reshape(-1, 1)
            
            if parsed_values_reshaped.shape[0] != original_values_column.shape[0]:
                print("Mismatch in the number of cells in parsed_values_reshaped and original_values_column.")
                return

            # Here, time_steps and wavelengths are the same length as parsed_values
            time_steps_reshaped = np.array(time_steps).reshape(-1, 1)
            wavelength = 190 + (position-1)*10  # Wavelength for the current column
            wavelengths_reshaped = np.array([wavelength]*len(time_steps)).reshape(-1, 1)

            # Concatenate parsed_values_reshaped, time_steps_reshaped, and wavelengths_reshaped along the second axis to form X:
            X = np.concatenate([parsed_values_reshaped, time_steps_reshaped, wavelengths_reshaped], axis=1)
            
            model_name = f'model_{position}.joblib'
            if os.path.exists(model_name) and load:
                model = load_model(model_name)
                print(f"********************Loading Neural Network for position {position} with loss = {best_losses[position]}")
                if relearn:
                    K.set_value(model.optimizer.learning_rate, lr)
            else:
                print(f"********************No existing model to load for position {position}. Starting with a fresh model.********************")
                model = create_model_complex()
            
            successful_epochs = 0
            unsuccessful_epochs = 0
            limit_reached = False


                        # Initialize a scaler for each column
            X_scaler_filename = f"X_scaler_{position}_new.joblib"
            y_scaler_filename = f"y_scaler_{position}_new.joblib"

            print(f"Statistics for position {position}:")

            # Flatten the array to 1D before calculating statistics
            original_values_column_1d = original_values_column.flatten()

            # # Compute statistics for the column of original_values
            # mean_original = np.mean(original_values_column_1d)
            # median_original = np.median(original_values_column_1d)
            # std_dev_original = np.std(original_values_column_1d)
            # skewness_original = skew(original_values_column_1d)
            # kurtosis_original = kurtosis(original_values_column_1d)
            # range_original = np.ptp(original_values_column_1d) #ptp stands for 'peak to peak'
            # Q1 = np.percentile(original_values_column_1d, 25)
            # Q3 = np.percentile(original_values_column_1d, 75)
            # IQR = Q3 - Q1        



            # # Print the statistics
            # print(f"Mean: {mean_original}")
            # print(f"Median: {median_original}")
            # print(f"Standard Deviation: {std_dev_original}")
            # print(f"Skewness: {skewness_original}")
            # print(f"Kurtosis: {kurtosis_original}")
            # print(f"Range: {range_original}")
            # print(f"1st Quartile: {Q1}")
            # print(f"3rd Quartile: {Q3}")
            # print(f"IQR: {IQR}")


            if os.path.exists(X_scaler_filename) and not load:
                # Load existing scalers
                X_scaler = joblib.load(X_scaler_filename)
                y_scaler = joblib.load(y_scaler_filename)
                # Transform the data
                X_scaled = X_scaler.transform(X)
                y_scaled = y_scaler.transform(original_values_column).ravel()
            else:
                # Initialize new scalers and fit-transform them on the data
                X_scaler = PowerTransformer(method='yeo-johnson').fit(X)
                y_scaler = PowerTransformer(method='yeo-johnson').fit(original_values_column)
                # Transform the data
                X_scaled = X_scaler.transform(X)
                y_scaled = y_scaler.transform(original_values_column).ravel()
                # Save the fitted scalers for later use
                joblib.dump(X_scaler, X_scaler_filename)
                joblib.dump(y_scaler, y_scaler_filename)
            

            counter = 0

            unscaled_loss = np.inf
            model_buffer = model

            while(unscaled_loss > 0.3):
                counter += 1
                print(f"Training model {position}, epoch {counter}(lr: {K.get_value(model.optimizer.learning_rate)}, best loss: {best_neural_loss}, rel loss: {relative_loss})")
                history = model.fit(X_scaled, y_scaled, batch_size=batch, epochs=1, verbose=1)
                current_loss = history.history['loss'][0]

                    # Calculate the loss in the original scale
                if counter % 10 == 0:
                    try:
                        y_pred_scaled = model_buffer.predict(X_scaled)
                        y_pred = y_scaler.inverse_transform(y_pred_scaled).ravel()  # unscale the predictions
                        unscaled_loss_new = np.mean(np.abs(original_values_column.ravel() - y_pred))
                        if not math.isnan(unscaled_loss_new):
                            print(f"Unscaled loss: {unscaled_loss}")
                            unscaled_loss = unscaled_loss_new
                    except Exception as e:
                        print(f"Can't predict unscaled loss due to error: {e}")

                if current_loss + 0.0001 < best_neural_loss:
                    best_neural_loss = current_loss
                    relative_loss = best_neural_loss
                    print(f"New best loss: {best_neural_loss}")
                    best_models[position] = model
                    model_buffer = model
                    successful_epochs += 1
                    unsuccessful_epochs = 0

                    if successful_epochs == save_iterations:
                        print(f"********************Saving Neural Network for column {position} after {successful_epochs} successful epochs.********************")
                        model.save(model_name)
                        best_models[position] = None
                        #best_losses = joblib.load("best_losses_alt4.joblib")
                        best_losses[position] = best_neural_loss
                        joblib.dump(best_losses, "best_losses_alt4.joblib")
                        successful_epochs = 0  # reset counter
                elif counter > 5:
                    if current_loss + 0.0001 < relative_loss:
                        relative_loss = current_loss
                        unsuccessful_epochs = 0
                        print(f"New best relative loss: {relative_loss}")
                    else:
                        unsuccessful_epochs += 1

                        if unsuccessful_epochs == early_stop_patience:
                            print(f"********************Stopping training for column {position} after {early_stop_patience} epochs without improvement********************")
                            unsuccessful_epochs = 0
                            break


                        if unsuccessful_epochs % reduce_lr_patience == 0:
                            old_lr = K.get_value(model.optimizer.learning_rate)
                            new_lr = old_lr * reduce_lr_factor
                            if new_lr > min_lr:
                                K.set_value(model.optimizer.learning_rate, new_lr)
                                print(f"********************Reduced learning rate from {old_lr} to {new_lr} after {reduce_lr_patience} epochs without improvement********************")
                            # elif old_lr > min_lr and not limit_reached:
                            #     print(f"********************Reduced learning rate from {old_lr} to {min_lr} after {reduce_lr_patience} epochs without improvement********************")
                            #     K.set_value(model.optimizer.learning_rate, min_lr)
                            #     limit_reached = True
                            else:
                                print(f"********************Can't reduce learning rate further — attempting increase********************")
                                K.set_value(model.optimizer.learning_rate, 0.2)
                                limit_reached = False
                                relative_loss = 5
            relative_loss = np.inf
            limit_reached = False

            if best_models[position] is not None:
                print("Saving current best model")
                model.save(model_name)
                #best_losses = joblib.load("best_losses_alt4.joblib")
                best_losses[position] = best_neural_loss
                joblib.dump(best_losses, "best_losses.joblib")

        print("Training completed")

    except (Exception, KeyboardInterrupt) as e:
        logging.error("An error occurred: %s", e)
        logging.exception("Full exception traceback:")
        print(f"Exception occurred: {e}")
        if best_models[position] is not None:
            print("Saving current best model")
            model.save(model_name)
            #best_losses = joblib.load("best_losses_alt4.joblib")
            best_losses[position] = best_neural_loss
            joblib.dump(best_losses, "best_losses_alt4.joblib")
        exit()


if __name__ == "__main__":
    main()
