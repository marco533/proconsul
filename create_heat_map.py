import csv
from turtle import end_fill
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

def compare_algorithms(algorithms_pair):

        df = pd.read_csv(f"tables/{algorithms_pair[0]}_vs_{algorithms_pair[1]}_f1_p2.csv")

        parameters = ["LCC_size", "Density", "Disgenes Percentage"]

        #This counter indicates the column where the first parameter is located
        column_counter = 2
        mode = ["absolute", "gradient"]

        for m in mode:

            for param in parameters:

                sorted_df = df.sort_values(by = param)
                data = sorted_df.values.tolist()

                #We convert our data to a convenient format
                data_array = np.array(data)
                yaxis = data_array[0,:]
                yaxis = np.delete(yaxis, [0,1,2,3,4,5])
                yaxis = ['KF Top 50', 'KF Top 100', 'KF Top 200', 'KF Top N', 'EX Top 50', 'EX Top 100', 'EX Top 200', 'EX Top N']
                xaxis = data_array[:,column_counter]
                xaxis = np.delete(xaxis, 0)
                data_array = np.delete(data_array,[0,1,2,3,4,5], axis=1)
                data_array = np.delete(data_array,0, axis=0)

                column_counter +=  1

                #This function generates a numerical matrix that is used for the heat map generation

                num_matrix = createMatrix(data_array, algorithms_pair, m)

                #Tranpose the matrix so the plot fits better on the screen
                num_matrix = np.transpose(num_matrix)

                #We plot the results
                cmap = plt.get_cmap('bwr')
                fig, ax = plt.subplots()
                img = ax.imshow(num_matrix, cmap = cmap)

                ax.set_xticks(np.arange(len(xaxis)))
                ax.set_yticks(np.arange(len(yaxis)))
                ax.set_yticklabels(yaxis)
                ax.set_xticklabels(xaxis)

                plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")
                plt.colorbar(img, shrink=0.5)
                plt.text(97,20, f"{algorithms_pair[1]} > {algorithms_pair[0]}")
                plt.text(96,-12, f"{algorithms_pair[0]} > {algorithms_pair[1]}")

                fig.suptitle(f"{param}: {algorithms_pair[0]} vs {algorithms_pair[1]}")

                # for i in range(len(yaxis)):
                #     for j in range(len(xaxis)):
                #         text = ax.text(j, i, num_matrix[i, j], ha="center", va="center", color="w")

                #plt.show()
                plt.savefig(f"heatmaps/{m}/{param}_{algorithms_pair[0]}_vs_{algorithms_pair[1]}.png")

                plt.close('all')


def createMatrix(data_array, algorithms_pair, mode):

    [rows, columns] = np.shape(data_array)
    num_matrix = np.zeros((rows, columns))

    for j in range(columns):
        for i in range(rows):
            if algorithms_pair[1] in data_array[i][j]:
                if mode == "absolute":
                    num_matrix[i][j] = -1
                else:
                    value = parseStringToFloat(data_array[i][j])
                    num_matrix[i][j] = value*(-1)

            elif 'draw' in data_array[i][j]:
                if mode == "absolute":
                    num_matrix[i][j] = 0
                else:
                    value = parseStringToFloat(data_array[i][j])
                    num_matrix[i][j] = value
            else:
                if mode == "absolute":
                    num_matrix[i][j] = 1
                else:
                    value = parseStringToFloat(data_array[i][j])
                    num_matrix[i][j] = value

    return num_matrix

def parseStringToFloat(str):

    str = str.replace(")", "")
    word,value = str.split(", ")
    value = float(value)

    return value

if __name__ == "__main__":

    algorithms = ["diamond","pdiamond","heat_diffusion"]

    algorithms_pair = [algorithms[0], algorithms[1]]

    compare_algorithms(algorithms_pair)

    algorithms_pair = [algorithms[0], algorithms[2]]

    compare_algorithms(algorithms_pair)


