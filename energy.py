########################################################################################
################ ENERGY FORECASTING USING LONG SHORT TERM MEMORY NETWORK  ##############
########################################################################################

import os
path = '/Users/lisa/Desktop/blue_sky/My Learnings/Machine Learning/Mastering Machine Learning/Projects/04 Deep Learning/LSTM_energy'
os.chdir(path)

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
# from keras import initializers
from keras.optimizers import SGD
from keras.models import load_model
from convert_supervised_learning import series_to_supervised
from math import sqrt
import glob as gb


############################### IMPORT & PREPARE DATASETS ##############################
########################################################################################

dataset_load = pd.read_csv('Load_history.csv')
dataset_load = dataset_load[~dataset_load['h1'].isnull() == True]

load = pd.melt(dataset_load, id_vars = ['zone_id','year','month','day'], var_name ='hour', value_name = 'load')
load['hour'] = load['hour'].str.replace('h',"")
load['load'] = load['load'].str.replace(",","")
load['date'] = pd.to_datetime(load[['year', 'month', 'day']])

dataset_temp = pd.read_csv("temperature_history.csv")
temp = pd.melt(dataset_temp, id_vars = ['station_id', 'year', 'month', 'day'], var_name = 'hour', value_name = 'temp')
temp['hour'] = temp['hour'].str.replace('h',"")
temp['date'] = pd.to_datetime(temp[['year', 'month', 'day']])
temp = temp.groupby(['date','hour']).agg({'temp':'mean'})
temp.reset_index(['date','hour'],inplace = True)

# Select zone 1 energy usage
dataset = load.loc[(load['zone_id'] == 1) & (load['date'] <= '2008-06-30'),:]
dataset = pd.merge(dataset, temp, on = ['date', 'hour'], how= 'left')
dataset =dataset[dataset['load'].isnull() ==False]
dataset = dataset.sort_values('date')

######################################## CHARTS ########################################
########################################################################################

# Chart of load over time
plt.rcParams["figure.figsize"] = [20, 6]
plt.plot(dataset['date'].astype('O'), dataset['load'].astype('float32'), linewidth = 0.5 )
plt.title('Energy load for zone 1')
plt.ylabel('Energy load')
plt.xlabel('Date')
plt.savefig("Energy Load Over Time")
plt.show()

# Chart of temperature over time
plt.rcParams["figure.figsize"] = [20, 6]
plt.plot(dataset['date'].astype('O'), dataset['temp'], color = 'green', linewidth = 0.5)
plt.title('Air Temperature')
plt.ylabel('Temp (Fahrenheit)')
plt.xlabel('Date')
plt.savefig("Air Temperature Over Time")
plt.show()

# Scatterplot of temperature vs load
plt.rcParams["figure.figsize"] = [20, 6]
plt.scatter(dataset['temp'], dataset['load'].astype('float'), color = 'grey', s = 2)
plt.ylabel('Energy load')
plt.xlabel("Temperature (Fahrenheit)")
plt.savefig("Energy Load vs Temperature")
plt.show()

# Check for extreme outliers 
plt.boxplot(dataset['load'].astype('float32'))
plt.title('Outlier Detection')
plt.ylabel('Energy load')
plt.savefig("Outlier Detection")
plt.show()

##################### DEFINE DATA PREPARATION AND MODEL FUNCTIONS ######################
########################################################################################

####### DEFINE FUNCTION:  FRAME DATASET INTO TIME STEP MATRIX ########

def timestep_matrix(timesteps, dataset):
    values = dataset[['load', 'year', 'month', 'day', 'hour', 'temp']].astype('float32').values     # Note: predicted output feature must be in column 1
    features = values.shape[1]
    
    # Normalise feature values
    scaler = preprocessing.MinMaxScaler(feature_range = (0, 1))
    scaled = scaler.fit_transform(values)
    
    # Reframe dataset into time-step sequences
    reframed = series_to_supervised(scaled, timesteps, 1)
    column_drop = (np.array(range(features))*-1)[1:features]
    reframed.drop(reframed.columns[column_drop], axis = 1, inplace = True)
    
    # Transform commands for single column arrays
    all_y = values[:,0]
    all_y = all_y.reshape(all_y.shape[0],1)
    scaled_y = scaler.fit_transform(all_y)
    # y = scaler.inverse_transform(scaled_y)

    return features, scaler, reframed, all_y, scaled_y


########### DEFINE FUNCTION: SPLIT TRAIN AND TEST SET ################

def train_test(reframed, test_obs):
    data_test = reframed[-test_obs:]             
    data_train = reframed[:-test_obs]

    return data_test, data_train


######### DEFINE FUNCTION: RESHAPE DATASETS FOR MODEL INPUT ##########

def reshape(data, timesteps, features):
    reframed_X = data.values[:,:-1]
    reshaped_Y = data.values[:,-1:]
    reshaped_X = reframed_X.reshape(reframed_X.shape[0], timesteps, features)
    return reshaped_X, reshaped_Y


########################## DEFINE THE MODEL ##########################

def lstm_model(timesteps, features, neurons, dropout, learning, momentum, decay, init, activation, activation_output, opt ) :
    model = Sequential()
    model.add(LSTM(neurons, input_shape = (timesteps, features), kernel_initializer = init, activation = activation))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation = activation_output))
    SGD( lr = learning, momentum = momentum, decay = decay, nesterov = False)
    model.compile(optimizer = opt, loss = 'mse') 
    return model


#################################### TUNE PARAMETERS ###################################
########################################################################################

#### Model input set-up #####
    
# Model parameters
neurons = 100
batch = 132
epochs = 2
dropout = 0.20
learning = 0.1
momentum = 0.8
decay = learning/epochs
activation = 'softsign'
activation_output = 'sigmoid'
init = 'glorot_uniform'
opt = 'sgd'


# Reframe and split dataset to input into LSTM model
test_obs = 30
timesteps = 7
repeats = 10

features, scaler, reframed, all_y, scaled_y = timestep_matrix(timesteps, dataset)
data_test, data_train = train_test(reframed, test_obs)

train_X, train_Y = reshape(data_train, timesteps, features)
test_X, test_Y = reshape(data_test, timesteps, features)


######################## DETERMINE NO OF EPOCHS ###################
# Grid search no of epochs to determine optimal no of epochs to use in model

epochs = [2, 5, 10]

for j in epochs:
    train = pd.DataFrame(); validation = pd.DataFrame()
    for i in range(repeats):
        model = lstm_model(timesteps, features, neurons, dropout, learning, momentum, decay, init, activation,activation_output, opt)
        history = model.fit(train_X, train_Y, validation_data = (test_X, test_Y), epochs = j, batch_size = batch, verbose =2)         
        train[i] = history.history['loss']
        validation[i] = history.history['val_loss']
 
    # Plot learning curve of train accuracy and test accuracy over epoch count. Then save for each run of epoch size
    chart_name = "Plot_epochs_" + str(j)  + ".png"
    plt.plot(train, color = 'blue', label = 'train')
    plt.plot(validation, color = 'orange', label = "validation")
    plt.title("No. of epochs: " + str(j))
    plt.xlabel("Epoch number")
    plt.ylabel("Loss")
    plt.savefig(chart_name)
    plt.show()

##### Optimal no of epochs to use:  10


###################### DETERMINE BATCH SIZE #####################
# Grid search batch size to determine optimal batch size to use in model

batches = [ 64, 32, 16, 8, 4, 1]

validation_all_batches = pd.DataFrame()
validation_all_batches_headers = []
k = -1
for j in batches:
    validation = pd.DataFrame()
    validation_average = []
    for i in range(repeats):
        model = lstm_model(timesteps, features, neurons, dropout, learning, momentum, decay, init, activation,activation_output, opt)
        history = model.fit(train_X, train_Y, validation_data = (test_X, test_Y), epochs = epochs, batch_size = j, verbose =2)         
        validation[i] = history.history['val_loss']
        validation_average.append(np.mean(list(validation[i])))
    k = k + 1    
    validation_all_batches[k] = validation_average
    validation_all_batches_headers.append(j)
    validation_all_batches.columns = validation_all_batches_headers
validation_average_all_batches = validation_all_batches.mean()
validation_stdev_all_batches = validation_all_batches.std()
print(validation_average_all_batches)
print(validation_stdev_all_batches)
plt.boxplot(validation_all_batches, labels = validation_all_batches.columns)
plt.show()

##### Optimal batch size to use:  10


######################## DETERMINE NO OF NEURONS ###################
# Grid search optimal no. of neurons in the first hidden layer

neurons = [10, 20, 40, 80, 100, 150]
           
validation_all_neurons = pd.DataFrame()
validation_all_neurons_headers = []
k = -1
for j in neurons:
    validation = pd.DataFrame()
    validation_average = []
    for i in range(repeats):
        model = lstm_model(timesteps, features, j, dropout, learning, momentum, decay, init, activation, activation_output, opt)
        history = model.fit(train_X, train_Y, validation_data = (test_X, test_Y), epochs = epochs, batch_size = batch, verbose =2)         
        validation[i] = history.history['val_loss']
        validation_average.append(np.mean(list(validation[i])))
    k = k + 1    
    validation_all_neurons[k] = validation_average
    validation_all_neurons_headers.append(j)
    validation_all_neurons.columns = validation_all_neurons_headers
validation_average_all_neurons = validation_all_neurons.mean()
validation_stdev_all_neurons = validation_all_neurons.std()
print(validation_average_all_neurons)
print(validation_stdev_all_neurons)
plt.boxplot(validation_all_neurons, labels = validation_all_neurons.columns)
plt.show()

# Optimal number of neurons is: 100    with loss: 0.003662


######################## DETERMINE BEST ACCTIVATION FUNCTION in HIDDEN LAYER ###################
# Grid search best activation function in the first hidden layer

activation_fns = ['softsign', 'relu', 'selu', 'sigmoid', 'hard_sigmoid', 'softplus', 'tanh', 'softmax', 'exponential','linear']
                  
validation_all_fns = pd.DataFrame()
validation_all_fns_headers = []
k = -1
for j in activation_fns:
    validation = pd.DataFrame()
    validation_average = []
    for i in range(repeats):
        model = lstm_model(timesteps, features, neurons, dropout, learning, momentum, decay, init, j, activation_output, opt)
        history = model.fit(train_X, train_Y, validation_data = (test_X, test_Y), epochs = epochs, batch_size = batch, verbose =2)         
        validation[i] = history.history['val_loss']
        validation_average.append(np.mean(list(validation[i])))
    k = k + 1    
    validation_all_fns[k] = validation_average
    validation_all_fns_headers.append(j)
    validation_all_fns.columns = validation_all_fns_headers
validation_average_all_fns = validation_all_fns.mean()
validation_stdev_all_fns = validation_all_fns.std()
print(validation_average_all_fns)
print(validation_stdev_all_fns)
plt.boxplot(validation_all_fns, labels = validation_all_fns.columns)
plt.show()


# Optimal activation function for hidden layer: selu


######################## DETERMINE BEST ACCTIVATION FUNCTION in OUTPUT LAYER ###################
# Grid search best activation function in the output layer

activation_fns = ['relu', 'sigmoid', 'hard_sigmoid', 'softplus', 'exponential']

validation_all_fns = pd.DataFrame()
validation_all_fns_headers = []
k = -1
for j in activation_fns:
    validation = pd.DataFrame()
    validation_average = []
    for i in range(repeats):
        model = lstm_model(timesteps, features, neurons, dropout, learning, momentum, decay, init, activation, j , opt)
        history = model.fit(train_X, train_Y, validation_data = (test_X, test_Y), epochs = epochs, batch_size = batch, verbose =2)         
        validation[i] = history.history['val_loss']
        validation_average.append(np.mean(list(validation[i])))
    k = k + 1    
    validation_all_fns[k] = validation_average
    validation_all_fns_headers.append(j)
    validation_all_fns.columns = validation_all_fns_headers
validation_average_all_fns = validation_all_fns.mean()
validation_stdev_all_fns = validation_all_fns.std()
print(validation_average_all_fns)
print(validation_stdev_all_fns)
plt.boxplot(validation_all_fns, labels = validation_all_fns.columns)
plt.show()

# Optimal activation function for output layer: relu


######################## DETERMINE BEST GRADIENT DESCENT ALGORITHM ###################
# Grid search best gradient descent algorithm 

gdas = ['adam', 'sgd', 'RMSprop', 'Adagrad', 'Adadelta', 'Adamax', 'Nadam']

validation_all_gdas = pd.DataFrame()
validation_all_gdas_headers = []
k = -1
for j in gdas:
    validation = pd.DataFrame()
    validation_average = []
    for i in range(repeats):
        model = lstm_model(timesteps, features, neurons, dropout, learning, momentum, decay, init, activation, activation_output , j)
        history = model.fit(train_X, train_Y, validation_data = (test_X, test_Y), epochs = epochs, batch_size = batch, verbose =2)         
        validation[i] = history.history['val_loss']
        validation_average.append(np.mean(list(validation[i])))
    k = k + 1    
    validation_all_gdas[k] = validation_average
    validation_all_gdas_headers.append(j)
    validation_all_gdas.columns = validation_all_gdas_headers
validation_average_all_gdas = validation_all_gdas.mean()
validation_stdev_all_gdas = validation_all_gdas.std()
print(validation_stdev_all_gdas)
print(validation_stdev_all_gdas)
plt.boxplot(validation_all_gdas, labels = validation_all_gdas.columns)
plt.show()

# Optimal gradient descent model: adam


######################## DETERMINE WEIGHT INITIALISER ###################
# Grid search optimal best weight initialisation

inits = ['glorot_uniform', 'glorot_normal', 'normal', 'he_normal', 'lecun_normal', 'he_uniform', 'uniform', 'random_uniform', 'random_normal', 'orthogonal', 'lecun_uniform', 'zero', 'one', 'identity']
      
validation_all_inits = pd.DataFrame()
validation_all_inits_headers = []
k = -1
for j in inits:
    validation = pd.DataFrame()
    validation_average = []
    for i in range(repeats):
        model = lstm_model(timesteps, features, neurons, dropout, learning, momentum, decay, j, activation, activation_output, opt)
        history = model.fit(train_X, train_Y, validation_data = (test_X, test_Y), epochs = epochs, batch_size = batch, verbose =2)         
        validation[i] = history.history['val_loss']
        validation_average.append(np.mean(list(validation[i])))
    k = k + 1    
    validation_all_inits[k] = validation_average
    validation_all_inits_headers.append(j)
    validation_all_inits.columns = validation_all_inits_headers
validation_average_all_inits = validation_all_inits.mean()
validation_stdev_all_inits = validation_all_inits.std()
print(validation_average_all_inits)
print(validation_stdev_all_inits)
plt.boxplot(validation_all_inits, labels = validation_all_inits.columns)
plt.show()

# Optimal best weight initialisation: random normal


######################## DETERMINE DROP-OUT RATE ###################
# Grid search optimal drop-out rate in the first hidden layer

dropouts = [0.1, 0.15, 0.2, 0.25]

validation_all_dropouts = pd.DataFrame()
validation_all_dropouts_headers = []
k = -1
for j in dropouts:
    validation = pd.DataFrame()
    validation_average = []
    for i in range(repeats):
        model = lstm_model(timesteps, features, neurons, j, learning, momentum, decay, init, activation, activation_output, opt)
        history = model.fit(train_X, train_Y, validation_data = (test_X, test_Y), epochs = epochs, batch_size = batch, verbose =2)         
        validation[i] = history.history['val_loss']
        validation_average.append(np.mean(list(validation[i])))
    k = k + 1    
    validation_all_dropouts[k] = validation_average
    validation_all_dropouts_headers.append(j)
    validation_all_dropouts.columns = validation_all_dropouts_headers
validation_average_all_dropouts = validation_all_dropouts.mean()
validation_stdev_all_dropouts = validation_all_dropouts.std()
print(validation_average_all_dropouts)
print(validation_stdev_all_dropouts)
plt.boxplot(validation_all_dropouts, labels = validation_all_dropouts.columns)
plt.show()


# Optimal drop-out rate: 0.15


######################## DETERMINE LEARNING RATE ###################
# Grid search optimal learning rate starting point

learnings = [0.4, 0.3, 0.2, 0.1, 0.05]

validation_all_lrs = pd.DataFrame()
validation_all_lrs_headers = []
k = -1
for j in learnings:
    validation = pd.DataFrame()
    validation_average = []
    for i in range(repeats):
        model = lstm_model(timesteps, features, neurons, dropout, j, momentum, decay, init, activation, activation_output, opt)
        history = model.fit(train_X, train_Y, validation_data = (test_X, test_Y), epochs = epochs, batch_size = batch, verbose =2)         
        validation[i] = history.history['val_loss']
        validation_average.append(np.mean(list(validation[i])))
    k = k + 1    
    validation_all_lrs[k] = validation_average
    validation_all_lrs_headers.append(j)
    validation_all_lrs.columns = validation_all_lrs_headers
validation_average_all_lrs = validation_all_lrs.mean()
validation_stdev_all_lrs = validation_all_lrs.std()
print(validation_average_all_lrs)
print(validation_stdev_all_lrs)
plt.boxplot(validation_all_lrs, labels = validation_all_lrs.columns)
plt.show()


# Optimal starting learning rate: 0.4


######################## DETERMINE MOMENTUM RATE ###################
# Grid search optimal momentum rate for the gradient descent algorithm

momentums = [0.75, 0.8, 0.85, 0.9, 0.95]

validation_all_mtms = pd.DataFrame()
validation_all_mtms_headers = []
k = -1
for j in momentums:
    validation = pd.DataFrame()
    validation_average = []
    for i in range(repeats):
        model = lstm_model(timesteps, features, neurons, dropout, learning, j, decay, init, activation, activation_output, opt)
        history = model.fit(train_X, train_Y, validation_data = (test_X, test_Y), epochs = epochs, batch_size = batch, verbose =2)         
        validation[i] = history.history['val_loss']
        validation_average.append(np.mean(list(validation[i])))
    k = k + 1    
    validation_all_mtms[k] = validation_average
    validation_all_mtms_headers.append(j)
    validation_all_mtms.columns = validation_all_mtms_headers
validation_average_all_mtms = validation_all_mtms.mean()
validation_stdev_all_mtms = validation_all_mtms.std()
print(validation_average_all_mtms)
print(validation_stdev_all_mtms)
plt.boxplot(validation_all_mtms, labels = validation_all_mtms.columns)
plt.show()


# Optimal momentum rate: 0.9


###################################### EVALUATE MODEL ##################################
########################################################################################

#### Model input set-up #####

# Model parameters
neurons = 100
batch = 10
epochs = 10
dropout = 0.15
learning = 0.4
momentum = 0.9
decay = learning/epochs
activation = 'selu'
activation_output = 'relu'
init = 'random_uniform'
opt = 'adam'


# Prepare data 
timesteps = 7
test_obs = 14
repeats = 10

features, scaler, reframed, all_y, scaled_y = timestep_matrix(timesteps, dataset)
data_test, data_train = train_test(reframed, test_obs)

train_X, train_Y = reshape(data_train, timesteps, features)
test_X, test_Y = reshape(data_test, timesteps, features)


############## EVALUATE MODELS USING OPTIMALLY TUNED PARAMETERS #############

accuracy_all = []
predicteds = pd.DataFrame()
k = 0
for i in range(repeats):
    model = lstm_model(timesteps, features, neurons, dropout, learning, momentum, decay, init, activation, activation_output, opt)
    model.fit(train_X, train_Y, validation_data = (test_X, test_Y), epochs = epochs, batch_size = batch, verbose =2)         
    predicted = model.predict(test_X)
    predicted_rescaled = scaler.inverse_transform(predicted)[0:,0].tolist()[1:]            # Predicted energy usage (test set date range). Predicts one time-step ahead (to compare with actual, remove the 1st value)

    actual = scaler.inverse_transform(test_Y)[:,0].tolist()[:-1]                            # Actual energy usage (to compare with predicted, remove the last value)
    
    rmse = sqrt(mean_squared_error(actual, predicted_rescaled)) / all_y.mean()   # ( mean_squared_error / no of observations ) / mean (actual output values)
    accuracy = 1 - rmse                                                         # Gives a zero to 1 accuracy measure
    accuracy_all.append(accuracy)
    
    k = k +1
    predicteds[k] = predicted_rescaled
    
# Calculate average of predictions and accuracies for all training runs
predicteds['mean'] = predicteds.iloc[:,0:k].mean(axis =1)             
average_accuracy = np.mean(accuracy_all)
std_accuracy = np.std(accuracy_all)

# Average accuracy: 96%


####### Chart of predicted load vs actual load in the test date range ##########
plt.plot(predicteds['mean'], color = 'blue', label = 'predicted')
plt.plot(actual, color = 'grey', label = 'actual')
plt.legend()
plt.show()



######################## TRAIN FINAL MODEL USING ENTIRE DATASET ########################
########################################################################################

#################################### TRAIN 1O MODELS ###################################
# Train on entire dataset

# Prepare whole data set for model input
data_all_X, data_all_Y = reshape(reframed, timesteps, features)

# Define the model
model_final = lstm_model(timesteps, features, neurons, dropout, learning, momentum, decay, init, activation, activation_output, opt)

no_models = 10
for m in range(no_models):
    model_final.fit(data_all_X, data_all_Y, epochs = epochs, batch_size = batch, verbose =2)         
    model_final_n = model_final.save('model_final_' + str(m) + '.h5')


################################## PREDICT & EVALUATE ####################################
# Predict energy load for whole data period, taking average predictions from all 10 models 
    
    
no_models = len(gb.glob(path + "/models/*"))
predicteds = pd.DataFrame()
accuracies = []
for m in range(no_models): 
    model_n = load_model(path+ '/models/model_final_' + str(m) + '.h5')
    predicted = model_n.predict(data_all_X)
    predicted_rescaled = scaler.inverse_transform(predicted)[:,0].tolist()[1:]             

    actual = scaler.inverse_transform(data_all_Y)[:,0].tolist()[:-1]
       
    rmse = (sqrt(mean_squared_error(actual, predicted_rescaled))) / all_y.mean()   
    accuracy = 1 - rmse   
    
    predicteds[m] = predicted_rescaled
    accuracies.append(accuracy)

predicteds['mean'] = predicteds.iloc[:,0:m].mean(axis=1)                                    
mean_accuracy = 1 - (sqrt(mean_squared_error(actual, predicteds['mean']))) / all_y.mean()   

print(predicteds['mean'])
print(mean_accuracy)

# Average accuracy: 86%


########################### CHART PREDICTED VALUES VS ACTUAL VALUES ####################
location = "load_zone_1"
dates = pd.Series(dataset['date'][8:])

plt.rcParams["figure.figsize"] = [20,10]
plt.plot(dates, actual, color = 'grey',  label = 'actual')
plt.plot(dates, predicteds['mean'], color = 'grey', alpha = 0.3, label = 'predicted')
plt.title("Predicted vs Actual Energy Load")
plt.ylabel('kwh')
plt.xlabel('Date')
plt.legend()
plt.savefig("Predicted vs Actual Energy Load")
plt.show()

