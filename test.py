# Installing and loading necessary packages and data files
from scipy import signal
import numpy as np
import time
import os
import sys
import matplotlib.pyplot as plt
# from google.colab import drive
import pandas as pd



drive_path = 'mobilize-tutorials-data-main/' # path to folder containing code and data
# drive_path = ' ' # path to folder containing code and data

# You can edit this first variable to choose which activity data you want to run (walk, run, bike, stairs)
# Options for dataset_name in the line below: [walk_sample.csv, run_sample.csv, stairs_sample.csv, bike_sample.csv]
dataset_name = 'walk_sample.csv'
data_path = drive_path+'energyExpenditureData/' + dataset_name
raw_IMU_data = np.loadtxt(data_path, delimiter=',')
print(raw_IMU_data.shape)

plot_start = 5000  # Sample number to start plotting from
plot_length = 500  # Number of samples to plot (seconds * 100)

plt.figure()
plt.title('Gyroscope signals on the shank')
plt.plot(raw_IMU_data[plot_start:plot_start + plot_length, 0:3])
plt.xlabel('Samples (100 Hz)')
plt.ylabel('Rate of rotation (deg/s)')
plt.legend(['x', 'y', 'z'])
plt.show()

plt.figure()
plt.title('Gyroscope signals on the thigh')
plt.plot(raw_IMU_data[plot_start:plot_start + plot_length, 3:6])
plt.xlabel('Samples (100 Hz)')
plt.ylabel('Rate of rotation (deg/s)')
plt.legend(['x', 'y', 'z'])
plt.show()

plt.figure()
plt.title('Accelerometer signals on the thigh')
plt.plot(raw_IMU_data[plot_start:plot_start + plot_length, 6:9])
plt.xlabel('Samples (100 Hz)')
plt.ylabel('Accelerations (m/s^2)')
plt.legend(['x', 'y', 'z'])
plt.show()

plt.figure()
plt.title('Accelerometer signals on the shank')
plt.plot(raw_IMU_data[plot_start:plot_start + plot_length, 9:12])
plt.xlabel('Samples (100 Hz)')
plt.ylabel('Accelerations (m/s^2)')
plt.legend(['x', 'y', 'z'])
plt.show()

# Design the filter
wn = 6 # Crossover frequency for low-pass filter (Hz)
filt_order = 4 # Fourth order filter
sample_freq = 100 # in Hz *Change this value if you use a different sample frequency*
b,a = signal.butter(filt_order, wn, fs=sample_freq) # params for low-pass filter

# Filter data
filtered_data = signal.filtfilt(b,a,raw_IMU_data[:,:-1], axis=0)

# Plot filtered and raw gyroscope shank z-axis signal
shank_gyro_z_index = 2 # column number for z-axis of shank gyroscope readings used to segment gait cycle

plt.figure()
plt.title('Gyroscope shank z-axis signal')
plt.plot(raw_IMU_data[plot_start:plot_start+plot_length,shank_gyro_z_index])
plt.plot(filtered_data[plot_start:plot_start+plot_length,shank_gyro_z_index])
plt.xlabel('Samples (100 Hz)')
plt.ylabel('Rate of rotation (deg/s)')
plt.legend(['raw','filtered'])
plt.show()

peak_height_thresh = 70  # minimum value of the shank IMU gyro-Z reading in deg/s
peak_min_dist = int(0.6 * sample_freq)  # min number of samples between peaks

# Find peaks
peak_index_list = \
signal.find_peaks(filtered_data[:, shank_gyro_z_index], height=peak_height_thresh, distance=peak_min_dist)[0]
peak_index_list_plot = \
signal.find_peaks(filtered_data[plot_start:plot_start + plot_length, shank_gyro_z_index], height=peak_height_thresh,
                  distance=peak_min_dist)[0]

# Plot
plt.figure()
plt.title('Gyroscope shank z-axis signal')
plt.plot(filtered_data[plot_start:plot_start + plot_length, shank_gyro_z_index])
axes = plt.gca()
ymin, ymax = axes.get_ylim()
xmin, xmax = axes.get_xlim()
plt.plot([xmin, xmax], [peak_height_thresh, peak_height_thresh], color='k', ls='--')
for i in peak_index_list_plot:
    plt.plot([i, i], [ymin, ymax], color='k')
plt.xlabel('Samples (100 Hz)')
plt.ylabel('Rate of rotation (deg/s)')
plt.legend(['filtered signal', 'threshold', 'detected heelstrikes'], loc='lower right')
plt.show()

# Variables to change depending on your data
weight = 68 # participant bodyweight in kg
height = 1.74 # participant height in m
num_bins = 30 # number of bins to discretize one gait cycle of data from each signal into a fixed size
stride_detect_window = 4*sample_freq # maximum time length for strides (seconds * samples/second)

# Initializing variables and useful parameters
gait_cnt = 0 # keep track of the number of gait cycles
gait_data = np.zeros((3+12*num_bins, len(peak_index_list)-1)) # storage for gait cycles of formatted data
deg2rad = 0.0174533 # rad / deg

# process time series data, format units, and discretize each gait cycle into a fixed number of discrete bins
def processRawGait(data_array, start_ind, end_ind, b, a, weight, height, num_bins=30):
    gait_data = data_array[start_ind:end_ind, :] # crop to the gait cycle
    gait_data = gait_data*np.array([deg2rad,-deg2rad,-deg2rad,deg2rad,-deg2rad,-deg2rad,1,-1,-1,1,-1,-1]) # flip y & z, convert to rad/s
    bin_gait = signal.resample(gait_data, num_bins, axis=0) # resample gait cycle of data into fixed number of bins along axis = 0 by default
    shift_flip_bin_gait = bin_gait.transpose() # get in shape of [feats x bins] for correct flattening
    model_input = shift_flip_bin_gait.flatten()
    model_input = np.insert(model_input, 0, [1.0, weight, height]) # adding a 1 for the bias term at start
    return model_input

for i in range(len(peak_index_list)-1): # looping through each gait cycle of data
    gait_start_index = peak_index_list[i] # index at start of gait
    gait_stop_index = peak_index_list[i+1] # index at end of gait
    if (gait_stop_index - gait_start_index) <= stride_detect_window: # if gait cycle within maximum time allowed
        gait_data[:,gait_cnt] = processRawGait(filtered_data, gait_start_index, gait_stop_index, b, a, weight, height, num_bins)
        gait_cnt += 1 # increment number of gait cycles of data stored
gait_data = gait_data[:,:gait_cnt] # get rid of empty storage

# Plotting discretized and continuous sampled data for one gait cycle
gait_cycle_to_plot = 1
plt.figure()
plt.title('Discretized and sampled data for gyroscope shank z-axis signal')
start_gait_index = peak_index_list[gait_cycle_to_plot]
stop_gait_index = peak_index_list[gait_cycle_to_plot+1]
plt.plot(filtered_data[start_gait_index:stop_gait_index,shank_gyro_z_index])

# plotting discretized data
gait_length = stop_gait_index - start_gait_index
x = np.linspace(start=0,stop=gait_length,num=num_bins+1)
bin = 2
plt.plot(x[:-1], gait_data[3+num_bins*bin:3+num_bins*(bin+1),gait_cycle_to_plot]*(-1/deg2rad))
plt.xticks(ticks=[0,gait_length],labels=[0,100])
plt.xlabel('Gait cycle (%)')
plt.ylabel('Rate of rotation (deg/s)')
plt.legend(['sampled data', 'discretized data'])
plt.show()

print("Format for processed gait data:",gait_data.shape)

# Load in model weights
model_dir = drive_path + 'energyExpenditureData/full_fold_aug/'  # path to model weights
model_weights = np.loadtxt(model_dir + 'weights.csv', delimiter=',')  # model weight vector
weights_unnorm = model_weights[3:]  # get rid of bias and height/weight offset
unnorm_weights = np.reshape(weights_unnorm, (-1, num_bins))

# Apply model to data to estimate energy expended
estimates = np.zeros(gait_cnt)
for i in range(gait_cnt):
    estimates[i] = np.dot(gait_data[:, i], model_weights)

# Plot energy estimates
plt.figure()
plt.plot(estimates)
plt.title("Energy expenditure per gait cycle")
plt.xlabel("Gait cycles")
plt.ylabel("Energy expenditure (W)")
plt.show()

# metabolics_breaths = np.loadtxt(drive_path+'energyExpenditureData/walking_metabolics.csv', delimiter=',', skiprows=1)


metabolics_breaths = pd.read_csv(drive_path+'energyExpenditureData/walking_metabolics.csv', skiprows=1)


avg_metabolics = np.mean(metabolics_breaths[len(metabolics_breaths)//2:])

plt.figure()
plt.plot(estimates)
plt.title("Energy expenditure per gait cycle")
plt.xlabel("Gait cycles")
plt.ylabel("Energy expenditure (W)")
plt.plot(np.linspace(start=0,stop=len(estimates),num=len(metabolics_breaths)),metabolics_breaths, c='k', alpha=0.5)
plt.plot([0,len(estimates)],[avg_metabolics, avg_metabolics], c='k')
plt.legend(['estimates per gait cycle', 'respirometry per breath','steady-state respirometry'])
avg_energy_expenditure_estimate = np.mean(estimates)
plt.plot([0,len(estimates)],[avg_energy_expenditure_estimate, avg_energy_expenditure_estimate], c='b')
plt.legend(['estimates per gait cycle', 'respirometry per breath','steady-state respirometry', 'steady-state estimates'])
plt.show()

# @title Plotting model weights
pelv_x_dir = " Z"# dir"
pelv_y_dir = " Y"# dir"
pelv_z_dir = " -X"# dir"
thigh_x_dir = " X"# dir"
thigh_y_dir = " -Z"#-dir"
thigh_z_dir = " Y"#-dir"
shank_x_dir = " X"#-dir"
shank_y_dir = " -Z"#-dir"
shank_z_dir = " Y"#-dir"
thigh = "Thigh"
shank = "Shank"
accel = " accel"
gryo = " gyro"

full_features = ['bias','weight','height',
                 shank+gryo+shank_x_dir,shank+gryo+shank_y_dir,shank+gryo+shank_z_dir,
                 thigh+gryo+thigh_x_dir,thigh+gryo+thigh_y_dir,thigh+gryo+thigh_z_dir,
                 thigh+accel+thigh_x_dir,thigh+accel+thigh_y_dir,thigh+accel+thigh_z_dir,
                 shank+accel+shank_x_dir,shank+accel+shank_y_dir,shank+accel+shank_z_dir]

plt.figure()
plt.imshow(unnorm_weights, cmap='RdBu')

xticks = [0,6.5,14,21.5,29]
yticks=[0,1,2,3,4,5,6,7,8,9,10,11]
gait_labels = ['0','25','50','75','100']
plt.yticks(ticks=yticks, labels=full_features[3:])
plt.xticks(ticks=xticks, labels=gait_labels)
plt.ylabel("Input signals")
plt.xlabel("Gait cycle (%)")
plt.title("Estimation model weights")
plt.show()

# @title Energy expenditure when transitioning between walking and running
walking_data_path = drive_path+'energyExpenditureData/walk_to_run_sample.csv'
raw_IMU_data = np.loadtxt(walking_data_path, delimiter=',')

b,a = signal.butter(filt_order, wn, fs=sample_freq) # params for low-pass filter
filtered_data = signal.filtfilt(b,a,raw_IMU_data[:,:-1], axis=0)
shank_gyro_z_index = 2 # column number for z-axis of shank gyroscope readings used to segment gait cycle

peak_index_list = signal.find_peaks(filtered_data[:,shank_gyro_z_index], height=peak_height_thresh, distance=peak_min_dist)[0]
peak_index_list_plot = signal.find_peaks(filtered_data[plot_start:plot_start+plot_length,shank_gyro_z_index], height=peak_height_thresh, distance=peak_min_dist)[0]

gait_data = np.zeros((3+12*num_bins, 1000)) # storage for up to 1000 gait cycles of formatted data
gait_cnt = 0
for i in range(len(peak_index_list)-1): # looping through each gait cycle of data
    gait_start_index = peak_index_list[i] # index at start of gait
    gait_stop_index = peak_index_list[i+1] # index at end of gait
    if (gait_stop_index - gait_start_index) <= stride_detect_window: # if gait cycle within maximum time allowed
        gait_data[:,gait_cnt] = processRawGait(filtered_data, gait_start_index, gait_stop_index, b, a, weight, height, num_bins)
        gait_cnt += 1 # increment number of gait cycles of data stored
gait_data = gait_data[:,:gait_cnt] # get rid of empty storage

estimates = np.zeros(gait_cnt)
for i in range(gait_cnt):
    estimates[i] = np.dot(gait_data[:,i], model_weights)

plt.figure()
plt.plot(estimates)
plt.plot([0,len(estimates)],[300,300])
plt.plot([0,len(estimates)],[750,750])

plt.title("Energy expenditure per gait cycle")
plt.xlabel("Gait cycles")
plt.ylabel("Energy expenditure (W)")
plt.legend(['Per step estimates','Steady-state walking energy','Steady-state running energy'])
plt.show()
