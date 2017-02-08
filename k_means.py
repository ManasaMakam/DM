#using pypyodbc to connect to sql server
import pypyodbc
from random import randint
import copy
#connection string for sql server local db
cnxn = pypyodbc.connect('DRIVER={SQL Server};SERVER=HARSHIT\SQLEXPRESS;DATABASE=DM_Assignment;UID=sa;PWD=sa')
#using cursor to read the tables
cursor = cnxn.cursor()
#getting all records from data which doesnt have missing values
cursor.execute("select * from DM_Assignment.dbo.clean_data")
data = cursor.fetchall()
A1 = []
A2 = []
A3 = []
A4 = []
A5 = []
A6 = []
A7 = []
A8 = []
A9 = []

for row in data:
  A1.append(row[0])
  A2.append(row[1])
  A3.append(row[2])
  A4.append(row[3])
  A5.append(row[4])
  A6.append(row[5])
  A7.append(row[6])
  A8.append(row[7])
  A9.append(row[8])

k = 5
th = 2
cent_move_final = th+1
cent_1 = []

for j in range(0,k):
  cent_2 = []
  cent_2.append(randint(min(A1), max(A1)))
  cent_2.append(randint(min(A2), max(A2)))
  cent_2.append(randint(min(A3), max(A3)))
  cent_2.append(randint(min(A4), max(A4)))
  cent_2.append(randint(min(A5), max(A5)))
  cent_2.append(randint(min(A6), max(A6)))
  cent_2.append(randint(min(A7), max(A7)))
  cent_2.append(randint(min(A8), max(A8)))
  cent_2.append(randint(min(A9), max(A9)))
  cent_1.append(cent_2)
i = 0
while cent_move_final > th and i < 10:
  
  i = i + 1
  cent_old = copy.deepcopy(cent_1)
  blocks = []
  for row in data:
    dist_1 = []
    for j in range(0,k):
      list_item = (((row[0] - cent_1[j][0]) ** 2) + ((row[1] - cent_1[j][1]) ** 2) + ((row[2] - cent_1[j][2]) ** 2) + ((row[3] - cent_1[j][3]) ** 2) + ((row[4] - cent_1[j][4]) ** 2) + ((row[5] - cent_1[j][5]) ** 2) + ((row[6] - cent_1[j][6]) ** 2) + ((row[7] - cent_1[j][7]) ** 2) + ((row[8] - cent_1[j][8]) ** 2) )** 0.5
      dist_1.append(list_item)
    blocks.append(dist_1.index(min(dist_1)))
  for j in range(0,k):
    for z in range(0,len(cent_1[0])):
      n = 0
      avg_2 = 0
      for l in range(0,len(blocks)):
        if blocks[l] == j:
          n = n + 1
          avg_2 = avg_2 + data[l][z]
  #        print "data point " + str(l)+ " "+ str(z) + "  " + str(data[l][z])
      if n !=0:
        avg_1 = avg_2/n
        cent_1[j][z] = avg_1
  cent_move_final = 0
  cent_move = 0
  for j in range(0,k):
    for z in range(0,len(cent_1[0])):
      cent_move = cent_move + ((cent_1[j][z] - cent_old[j][z])**2)
    cent_move_final = cent_move_final + (cent_move ** 0.5)

num_tot_1 = []

for j in range(0,k):
  num_ben = 0
  num_mal = 0
  num_tot = []
  for z in range(len(blocks)):
    if blocks[z] == j:
      if data[z][9] == 2:
        num_ben = num_ben + 1
      else:
        num_mal = num_mal + 1
  num_tot.append(num_ben)
  num_tot.append(num_mal)
  num_tot_1.append(num_tot)

blocks_ben_mal = []
for count in num_tot_1:
  if count[0] > count[1]:
    blocks_ben_mal.append("benign")
  else:
    blocks_ben_mal.append("malignant")
blocks_error = []
for j in range(0,k):
  if blocks_ben_mal[j] == "benign":
    blocks_error.append(round(100*float(num_tot_1[j][1])/(num_tot_1[j][0]+num_tot_1[j][1]),2))
  else:
    blocks_error.append(round(100*float(num_tot_1[j][0])/(num_tot_1[j][0]+num_tot_1[j][1]),2))
errors = []
error_rate = 0
for j in range(0,k):
  temp = []
  temp.append(blocks_ben_mal[j])
  temp.append(blocks_error[j])
  errors.append(temp)
  error_rate = error_rate + blocks_error[j]
text_file = open("DM_Assignment_output.txt", "a")
text_file.write("\n"+ str(error_rate)+ "," + str(k) )
text_file.close()

  
  
  
  
  



  











