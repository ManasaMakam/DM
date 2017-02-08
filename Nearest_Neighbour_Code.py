#using pypyodbc to connect to sql server
import pypyodbc

#connection string for sql server local db
cnxn = pypyodbc.connect('DRIVER={SQL Server};SERVER=HARSHIT\SQLEXPRESS;DATABASE=DM_Assignment;UID=sa;PWD=sa')
#using cursor to read the tables
cursor = cnxn.cursor()
#getting all records from data which doesnt have missing values
cursor.execute("select * from DM_Assignment.dbo.Cancer_non_missing")
non_missing = cursor.fetchall()
#getting all records from data which has missing values
cursor.execute("select * from DM_Assignment.dbo.missing_values")
missing_values = cursor.fetchall()
#looping through every record in missing data to find closest neighbour
for miss_row in missing_values:
  dist = []
  #looping through every record in non missing data 
  for no_miss_row in non_missing:
    #Variable 10 is benign/malignant classification. Making sure we compare benign with benign and malignant data with malignant
    if miss_row[10] == no_miss_row[10]:
      #calculating distance between two points
      dist_item = ((miss_row[1] - no_miss_row[1]) ** 2) + ((miss_row[2] - no_miss_row[2]) ** 2) + ((miss_row[3] - no_miss_row[3]) ** 2) + ((miss_row[4] - no_miss_row[4]) ** 2) + ((miss_row[5] - no_miss_row[5]) ** 2) + ((miss_row[7] - no_miss_row[7]) ** 2) + ((miss_row[8] - no_miss_row[8]) ** 2) + ((miss_row[9] - no_miss_row[9]) ** 2) 
    else: 
      #setting the distance to an arbitarily high number as a placeholder
      dist_item = 10000
    #appending all the distances to a list, to find the index of minimum distance
    dist.append(dist_item)
  #using the index of minimum distance, find that row in non missing data
  temp = non_missing[dist.index(min(dist))]
  #generating sql statement which updates A7 (missing) value with A7 value of the closest neighbour 
  sql = "UPDATE DM_Assignment.dbo.missing_values SET A7 = '" + str(temp[6]) + "' WHERE SCN = " + str(miss_row[0]) + " and A7 = '?'"
  cursor.execute(sql)
  cnxn.commit()







  
