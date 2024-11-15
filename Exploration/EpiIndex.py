import mne
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
import networkx as nx
from scipy import signal
import seaborn as sns



def get_EI(raw, window_size=5000, overlap=1,lambda_=125):
    #Making a matrix of U_n, where the rows are every channel and the columns are the windows
    #Getting the channels
    channels = raw.ch_names
    #Getting the number of channels
    n_channels=len(channels)
    #Getting the number of windows
  
    n_windows=len(range(0, len(raw.get_data()[0]) - window_size + 1, overlap))
    #Making a matrix of zeros
    U_n_matrix=np.zeros((n_channels,n_windows))
    ER_matrix=np.zeros((n_channels,n_windows))
    ER_n_array=np.zeros(n_channels)
    #Alarm time array
    alarm_time=np.zeros(n_channels)
    #Detection time array
    #detection_time=np.zeros(n_channels)
    #A loop for every channel
    for k in range(n_channels):
        #Getting the data of the channel
        data=raw.get_data()[k]
        #Getting the ER values
        time_points, ER_values = calculate_ER(data,raw, window_size, overlap)
        ER=ER_values
        #Normalizing between 0 and 1
        ER=(ER-np.min(ER))/(np.max(ER)-np.min(ER))
        ER_matrix[k,:]=ER
        N=len(ER)
        ER_n=(1/N)*np.sum(ER)
        #Getting the ER_n values in the 
        ER_n_array[k]=ER_n

        ##Getting U_n
        U_n=np.zeros(len(ER))
        v=0.1
        u_min=0
        alarm_times=[]
        for i in range(N):
            U_n[i]=np.sum(ER[0:i]-ER_n-0.1)
            u_min=np.min(U_n)
            if (U_n[i]-u_min)>lambda_:
                #print('Anomaly detected at window number ',i, ' for channel ',k)    
                alarm_times.append(i)
                u_min=0
                U_n[i]=0
        #Saving the U_n values in the matrix
        U_n_matrix[k,:]=U_n
        #Getting the alarm time
        try:
            alarm_time[k]=alarm_times[0]
            print(f'Alarm time for channel {k} is {alarm_time[k]}')
        except:
            alarm_time[k]=0
            print(f'No alarm time detected for channel {k}')
            print('Setting alarm time to 0, please check the lambda value')

            print('-------------------')
        
        
    #Getting EI
    N0=np.min(alarm_time)
    Ei=[]
    tau=1
    fs=raw.info['sfreq']
    #H variable is equal to 5 seconds, so 
    H=5*fs
    #sum from detection time to the end of the signal
    for k in range(n_channels):
        Ei.append(((1/(alarm_time[k]-N0+tau))*np.sum(ER_matrix[k,int(alarm_time[k]):int(alarm_time[k]+H)])))

    Ei_n=Ei/np.max(Ei)

    return Ei_n, ER_matrix, U_n_matrix, ER_n_array, alarm_time


def calculate_ER(data,raw, window_size, overlap):
    """
    Calculate time-varying Energy Ratio (ER) from Theta(w) using a sliding window.

    Parameters:
    - S: Periodogram estimate for Theta(w)
    - f: Frequency vector
    - window_size: Size of the sliding window in samples
    - overlap: Overlap between consecutive windows in samples

    Returns:
    - time_points: Array of time points corresponding to the center of each window
    - ER_values: Array of time-varying ER values
    """

    # Initialize empty arrays to store results
    time_points = []
    ER_values = []

    # Iterate through the signal with the sliding window
    for start in range(0, len(data) - window_size + 1, overlap):
        end = start + window_size
        #(f, S)= signal.welch(data[start:end], fs=raw.info['sfreq'], nperseg=1024*5)
        (f,S)=signal.periodogram(data[start:end],fs=raw.info['sfreq'],scaling='density')

        # Calculate energy in each frequency band for the current window
        ETheta = np.sum(S[np.where((f >= 3.5) & (f < 7.4))])
        EAlpha = np.sum(S[np.where((f >= 7.4) & (f < 12.4))])
        EBeta = np.sum(S[np.where((f >= 12.4) & (f < 24))])
        EGamma = np.sum(S[np.where((f >= 24) & (f <= 97))])

        # Calculate Energy Ratio (ER) for the current window
        ER = (EBeta + EGamma) / (ETheta + EAlpha)

        # Store results
        time_points.append((start + end) / 2)  # Use the center of the window as the time point
        ER_values.append(ER)

    return np.array(time_points), np.array(ER_values)


def calculate_optimal_lambda_vs1(ER,lambdas_):
    ER=(ER-np.min(ER))/(np.max(ER)-np.min(ER))
    N=len(ER)
    ER_n=(1/N)*np.sum(ER)
    

    U_n=np.zeros(len(ER))

    #Lambda is an array to explore the best value for the threshold
    alarm_times=np.arange(0,len(ER),1)
    best_lambda=0

    for lambda_ in lambdas_:
        U_n_temp=np.zeros(len(ER))
        v=0.1   
        u_min=0 
        temp_alarm_times=[]
        print('lambda is ',lambda_)
        for i in range(N):
            U_n_temp[i]=np.sum(ER[0:i]-ER_n-0.1)
            u_min=np.min(U_n_temp)
            if (U_n_temp[i]-u_min)>lambda_:
                #print('Anomaly detected at window number ',i)
                temp_alarm_times.append(i)
                u_min=0
                U_n_temp[i]=0
        #We are trying to find the lambda which give the least amount of alamrm time, still >0
        print ('Total alarm times are ',len(temp_alarm_times))
        print ('Current best lambda is ',best_lambda)
        print ('Current best alarm times are ',len(alarm_times))
        print('-------------------')
        if len(temp_alarm_times)<len(alarm_times) and len(temp_alarm_times)>0:
            alarm_times=temp_alarm_times
            best_lambda=lambda_
            U_n=U_n_temp
    return best_lambda





def get_EI_optimal_lambda_vs1(raw,lambdas_, window_size=5000, overlap=1):
    #Making a matrix of U_n, where the rows are every channel and the columns are the windows
    #Getting the channels
    channels = raw.ch_names
    #Getting the number of channels
    n_channels=len(channels)
    #Getting the number of windows  
    n_windows=len(range(0, len(raw.get_data()[0]) - window_size + 1, overlap))
    #Making a matrix of zeros
    best_lambdas=np.zeros(n_channels)
    for k in range(n_channels):
        #Getting the data of the channel
        data=raw.get_data()[k]
        #Getting the ER values
        time_points, ER_values = calculate_ER(data,raw, window_size, overlap)
        ER=ER_values
        #Normalizing between 0 and 1
        ER=(ER-np.min(ER))/(np.max(ER)-np.min(ER))
        N=len(ER)
        print('Calculating optimal lambda for channel ',k)
        optimal_lambda=calculate_optimal_lambda_vs1(ER,lambdas_)
        print('---> Optimal lambda for channel ',k,' is ',optimal_lambda)
        print('-------------------')
        best_lambdas[k]=optimal_lambda

    global_lambda=np.min(best_lambdas)

    Ei_n, ER_matrix, U_n_matrix, ER_n_array, alarm_time=get_EI(raw, window_size, overlap,global_lambda)
    return Ei_n, ER_matrix, U_n_matrix, ER_n_array, alarm_time



def get_ei_optimal_vs2(raw, window_size=5000, overlap=1):
    #Making a matrix of U_n, where the rows are every channel and the columns are the windows
    #Getting the channels
    channels = raw.ch_names
    #Getting the number of channels
    n_channels=len(channels)
    #Getting the number of windows
  
    n_windows=len(range(0, len(raw.get_data()[0]) - window_size + 1, overlap))
    #Making a matrix of zeros
    U_n_matrix=np.zeros((n_channels,n_windows))
    ER_matrix=np.zeros((n_channels,n_windows))
    ER_n_array=np.zeros(n_channels)
    derivates_d1_matrix=np.zeros((n_channels,n_windows-1))
    derivates_d2_matrix=np.zeros((n_channels,n_windows-2))
    #Alarm time array
    alarm_time=np.zeros(n_channels)

    #Detection time array
    #detection_time=np.zeros(n_channels)
    #A loop for every channel
    for k in range(n_channels):
        #Getting the data of the channel
        data=raw.get_data()[k]
        #Getting the ER values
        time_points, ER_values = calculate_ER(data,raw, window_size, overlap)
        ER=ER_values
        #Normalizing between 0 and 1
        ER=(ER-np.min(ER))/(np.max(ER)-np.min(ER))
        ER_matrix[k,:]=ER
        N=len(ER)
        ER_n=(1/N)*np.sum(ER)
        #Getting the ER_n values in the 
        ER_n_array[k]=ER_n

        #First derivative of ER
        dt=1
        d_ER=np.diff(ER)/dt
        derivates_d1_matrix[k,:]=d_ER
        #Second derivative of ER
        d2_ER=np.diff(d_ER)/dt
        derivates_d2_matrix[k,:]=d2_ER
        alarm_time[k]=np.argmax(d2_ER)
        print(f'Alarm time for channel {k} is {alarm_time[k]}')        
    #Getting EI
    N0=np.min(alarm_time)
    Ei=[]
    tau=1
    fs=raw.info['sfreq']
    #H variable is equal to 5 seconds, so 
    H=5*fs
    #sum from detection time to the end of the signal
    for k in range(n_channels):
        Ei.append(((1/(alarm_time[k]-N0+tau))*np.sum(ER_matrix[k,int(alarm_time[k]):int(alarm_time[k]+H)])))
    Ei_n=Ei/np.max(Ei)
    return Ei_n, ER_matrix, U_n_matrix, ER_n_array, alarm_time, derivates_d1_matrix, derivates_d2_matrix



#Function for ploting matriz from EI
# def plotting_ei(Ei_n, ER_matrix,channels, derivatives_d1=None):
#     if derivatives_d1 is None:
#         plt.imshow(ER_matrix,cmap='viridis',interpolation='bicubic',aspect='auto',extent=[0,40000,0,22])
#         #colorbar
#         plt.colorbar()
#         plt.yticks(np.arange(len(channels)), channels)
#         plt.xlabel('Window number')
#         plt.ylabel('Channel name')
#         plt.title('ER_n')
#         plt.show()

#         #Plting a barplt of the EI values for every channel
#         plt.bar(channels,Ei_n)
#         plt.xlabel('Channel name')
#         plt.ylabel('EI')
#         plt.show()
#     else:
#         plt.imshow(ER_matrix,cmap='viridis',interpolation='bicubic',aspect='auto',extent=[0,40000,0,22])
#         #colorbar
#         plt.colorbar()
#         plt.yticks(np.arange(len(channels)), channels)
#         plt.xlabel('Window number')
#         plt.ylabel('Channel name')
#         plt.title('ER_n')
#         plt.show()

#         #Plting a barplt of the EI values for every channel
#         plt.bar(channels,Ei_n)
#         plt.xlabel('Channel name')
#         plt.ylabel('EI')
#         plt.show()

#         plt.imshow(derivatives_d1,cmap='viridis',interpolation='bicubic',aspect='auto',extent=[0,40000,0,22])
#         #colorbar
#         plt.colorbar()
#         plt.yticks(np.arange(len(channels)), channels)
#         plt.xlabel('Window number')
#         plt.ylabel('Channel name')
#         plt.title('ER_n')
#         plt.show()



def plotting_ei(Ei_n, ER_matrix, channels, derivatives_d1=None, save_path=None):
    if derivatives_d1 is None:
        plt.imshow(ER_matrix, cmap='viridis', interpolation='bicubic', aspect='auto', extent=[0, 40000, 0, 22])
        plt.colorbar()
        plt.yticks(np.arange(len(channels)), channels)
        plt.xlabel('Window number')
        plt.ylabel('Channel name')
        plt.title('ER_n')
        if save_path:
            plt.savefig(save_path + '_ER_matrix.png')  # Save ER_matrix plot
        else:
            plt.show()

        plt.figure(figsize=(20, 5))
        plt.bar(channels, Ei_n)
        plt.xlabel('Channel name')
        plt.xticks(rotation=90)
        plt.ylabel('EI')
        if save_path:
            plt.savefig(save_path + '_barplot.png')  # Save barplot
        else:
            plt.show()
    else:
        plt.imshow(ER_matrix, cmap='viridis', interpolation='bicubic', aspect='auto', extent=[0, 40000, 0, 22])
        plt.colorbar()
        plt.yticks(np.arange(len(channels)), channels)
        plt.xlabel('Window number')
        plt.ylabel('Channel name')
        plt.title('ER_n')
        if save_path:
            plt.savefig(save_path + '_ER_matrix.png')  # Save ER_matrix plot
        else:
            plt.show()

        plt.bar(channels, Ei_n)
        plt.xlabel('Channel name')
        plt.ylabel('EI')
        if save_path:
            plt.savefig(save_path + '_barplot.png')  # Save barplot
        else:
            plt.show()

        plt.imshow(derivatives_d1, cmap='viridis', interpolation='bicubic', aspect='auto', extent=[0, 40000, 0, 22])
        plt.colorbar()
        plt.yticks(np.arange(len(channels)), channels)
        plt.xlabel('Window number')
        plt.ylabel('Channel name')
        plt.title('ER_n')
        if save_path:
            plt.savefig(save_path + '_derivatives.png')  # Save derivatives plot
        else:
            plt.show()
