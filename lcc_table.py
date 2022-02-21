#
# This script makes comparisons among diamond and prob results and outputs best algorithm per each LCC size
# and per each category (extended and kfold, top 50, 100, 200 , N ) 
#


import csv
from utils.data_utils import string_to_filename


#this function takes one LCC group and return the line to print on the final table (line composed by 8 values)
def manage_one_group(diseases):

    with open('tmp_table.csv', 'w') as of:

        header = ['kf_top_50', 'kf_top_100', 'kf_top_200','kf_top_N','ev_top_50', 'ev_top_100', 'ev_top_200','ev_top_N']

        writer = csv.writer(of)    
        writer.writerow(header)
    
        data=[]

        #read from the four files (diamond and prob diamond, kfold and extended)
        for disease in diseases:   

            rows1 = []
            disease = string_to_filename(disease)
            with open("results/kfold/diamond_on_"+disease+"_kfold.csv", 'r') as file1:
                csvreader1 = csv.reader(file1)
                header = next(csvreader1)
                for row in csvreader1:
                    rows1.append(row)

            rows2 = []
            disease = string_to_filename(disease)
            with open("results/kfold/prob_diamond_on_"+disease+"_kfold.csv", 'r') as file2:
                csvreader2 = csv.reader(file2)
                header = next(csvreader2)
                for row in csvreader2:
                    rows2.append(row)


            rows3 = []
            disease = string_to_filename(disease)            
            with open("results/extended/diamond_on_"+disease+"_extended.csv", 'r') as file3:
                csvreader3 = csv.reader(file3)
                header = next(csvreader3)
                for row in csvreader3:
                    rows3.append(row)

            rows4 = []
            disease = string_to_filename(disease)
            with open("results/extended/prob_diamond_on_"+disease+"_extended.csv", 'r') as file4:
                csvreader4 = csv.reader(file4)
                header = next(csvreader4)
                for row in csvreader4:
                    rows4.append(row)


            #comparisons among read values to get best algorithm

            #KFOLD
            #Top 50

            #parsing of the desired value
            rows1[2][1] = float((rows1[2][1].strip("(").strip(")").split(","))[0])
            rows2[2][1] = float((rows2[2][1].strip("(").strip(")").split(","))[0])
            if rows1[2][1]>rows2[2][1]: 
                kf_top_50_to_print = "diamond"
            elif rows1[2][1]<rows2[2][1]:
                kf_top_50_to_print = "prob diamond"
            else:
                kf_top_50_to_print = "same"

            # Top 100

            #parsing of the desired value            
            rows1[2][2] = float((rows1[2][2].strip("(").strip(")").split(","))[0])
            rows2[2][2] = float((rows2[2][2].strip("(").strip(")").split(","))[0])            
            if rows1[2][2]>rows2[2][2]: 
                kf_top_100_to_print = "diamond"
            elif rows1[2][2]<rows2[2][2]:
                kf_top_100_to_print = "prob diamond"
            else:
                kf_top_100_to_print = "same"

            # Top 200

            #parsing of the desired value
            rows1[2][3] = float((rows1[2][3].strip("(").strip(")").split(","))[0])
            rows2[2][3] = float((rows2[2][3].strip("(").strip(")").split(","))[0])            
            if rows1[2][3]>rows2[2][3]: 
                kf_top_200_to_print = "diamond"
            elif rows1[2][3]<rows2[2][3]:
                kf_top_200_to_print = "prob diamond"
            else:
                kf_top_200_to_print = "same"

            # Top N

            #parsing of the desired value
            rows1[2][4] = float((rows1[2][4].strip("(").strip(")").split(","))[0])
            rows2[2][4] = float((rows2[2][4].strip("(").strip(")").split(","))[0])
            if rows1[2][4]>rows2[2][4]: 
                kf_top_N_to_print = "diamond"
            elif rows1[2][4]<rows2[2][4]:
                kf_top_N_to_print = "prob diamond"
            else:
                kf_top_N_to_print = "same"


            #EXTENDED
            #Top 50

            #parsing of the desired value
            rows3[2][1] = float(rows3[2][1])
            rows4[2][1] = float(rows4[2][1])
            if rows3[2][1]>rows4[2][1]: 
                ev_top_50_to_print = "diamond"
            elif rows3[2][1]<rows4[2][1]:
                ev_top_50_to_print = "prob diamond"
            else:
                ev_top_50_to_print = "same"

            # Top 100

            #parsing of the desired value
            rows3[2][2] = float(rows3[2][2])
            rows4[2][2] = float(rows4[2][2])
            if rows3[2][2]>rows4[2][2]: 
                ev_top_100_to_print = "diamond"
            elif rows3[2][2]<rows4[2][2]:
                ev_top_100_to_print = "prob diamond"
            else:
                ev_top_100_to_print = "same"

            # Top 200

            #parsing of the desired value
            rows3[2][3] = float(rows3[2][3])
            rows4[2][3] = float(rows4[2][3])
            if rows3[2][3]>rows4[2][3]: 
                ev_top_200_to_print = "diamond"
            elif rows3[2][3]<rows4[2][3]:
                ev_top_200_to_print = "prob diamond"
            else:
                ev_top_200_to_print = "same"

            # Top N
            
            #parsing of the desired value
            rows3[2][4] = float(rows3[2][4])
            rows4[2][4] = float(rows4[2][4])
            if rows3[2][4]>rows4[2][4]: 
                ev_top_N_to_print = "diamond"
            elif rows3[2][4]<rows4[2][4]:
                ev_top_N_to_print = "prob diamond"
            else:
                ev_top_N_to_print = "same"


            riga=[kf_top_50_to_print,kf_top_100_to_print,kf_top_200_to_print,kf_top_N_to_print,  ev_top_50_to_print,ev_top_100_to_print,ev_top_200_to_print,ev_top_N_to_print]
            data.append(riga)
        writer.writerows(data)


    #now on tmp_table.csv there are 8 values per row, and a row per each disease of the group

    #for each of the 8 columns get the most frequent algorithm and put it in row_to_print
    with open('tmp_table.csv', 'r') as rf:

        row_to_print =[]
        rows=[]
        reader = csv.reader(rf)
        header = next(reader)    
        for row in reader:
            rows.append(row)
        for i in range(8):
            column=[]
            for j in range(len(rows)):
                column.append(rows[j][i])
            lcc_value = max(column,key=column.count)
            row_to_print.append(lcc_value)
    return row_to_print  #return the 8 values: 8 best algorithms per each category in this group

      

if __name__ == "__main__":

    #groups of diseases splitted by LCC size
    LCC_50_100 = ['Osteoporosis, Senile','Steatohepatitis']
    LCC_100_150 = ['Adenocarcinoma, Basal Cell','Heart Failure, Right-Sided']
    LCC_150_200 = ['Liver carcinoma']

    rows_to_print=[]

    i=1
    for group in [LCC_50_100,LCC_100_150,LCC_150_200]:

        #get name of the group te be printed on final table
        if i==1:        
            group_name = f'{LCC_50_100=}'.split('=')[0]
        if i==2:
            group_name = f'{LCC_100_150=}'.split('=')[0]
        i+=1

        #row_to_print = each row, one per group
        #rows_to_print = all the rows to print on the table
        #call core function
        row_to_print = manage_one_group(group)  
        row_to_print.insert(0,group_name)
        rows_to_print.append(row_to_print)

    #open and write on the final table (each row per one LCC group)
    with open('table.csv', 'w') as wf:

        header = ['LCC group','kf_top_50', 'kf_top_100', 'kf_top_200','kf_top_N','ev_top_50', 'ev_top_100', 'ev_top_200','ev_top_N']

        writer = csv.writer(wf)    
        writer.writerow(header)
        writer.writerows(rows_to_print)
