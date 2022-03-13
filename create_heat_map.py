import csv
from turtle import end_fill
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np


def createMatrix(data_array):

    [rows, columns] = np.shape(data_array)
    num_matrix = np.zeros((rows, columns))

    for j in range(columns):
        for i in range(rows):
            if 'heat' in data_array[i][j]:
                num_matrix[i][j] = 0
            elif 'draw' in data_array[i][j]:
                num_matrix[i][j] = 1
            else:
                num_matrix[i][j] = 2

    return num_matrix

if __name__ == "__main__":

    #We open the disease table file to sort it
    df = pd.read_csv('tables/diamond_vs_heat_diffusion_f1_p3.csv')
    sorted_df = df.sort_values(by=["LCC_size"])
    data = sorted_df.values.tolist()
    
    #We convert our data to a convenient format
    data_array = np.array(data)
    #yaxis = data_array[0,:]
    #yaxis = np.delete(yaxis, [0,1,2,3,4,5])
    yaxis = ['KF Top 50', 'KF Top 100', 'KF Top 200', 'KF Top N', 'EX Top 50', 'EX Top 100', 'EX Top 200', 'EX Top N']
    xaxis = data_array[:,2]
    xaxis = np.delete(xaxis, 0)
    data_array = np.delete(data_array,[0,1,2,3,4,5], axis=1)
    data_array = np.delete(data_array,0, axis=0)
    #print(xaxis)
    #This function generates a numerical matrix that can be used for the heat map generation
    print(data_array)
    num_matrix = createMatrix(data_array)
    
    #Tranpose the matrix so the plot fits better on the screen
    num_matrix = np.transpose(num_matrix)
    #We plot the results
    cmap = plt.get_cmap('bwr')
    fig, ax = plt.subplots()
    ax.imshow(num_matrix, cmap = cmap)
    ax.set_xticks(np.arange(len(xaxis)))
    ax.set_yticks(np.arange(len(yaxis)))
    ax.set_yticklabels(yaxis)
    ax.set_xticklabels(xaxis)
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")
    
    fig.suptitle('LCC_size: Diamond (red) vs Heat diffusion (blue)')

    # for i in range(len(yaxis)):
    #     for j in range(len(xaxis)):
    #         text = ax.text(j, i, num_matrix[i, j], ha="center", va="center", color="w")
    
    plt.show()
    
    plt.close('all')

   
