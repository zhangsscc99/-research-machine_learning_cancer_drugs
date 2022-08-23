# %%
#a code block that deals with concentration and drug names
#we do not use this block for now.

import pandas as pd
f1 = pd.read_csv('B.csv')
f2 = pd.read_csv('C.csv')
f3=pd.read_csv('D.csv')
f4=pd.read_csv('E.csv')
f5=pd.read_csv('A.csv')
file = [f1,f2,f3,f5]
train = pd.concat(file,axis=1)
train.to_csv("train.csv", index=0, sep=',')


# %%
#do some further processing with "A" because there are some data error in it
def remove_extradots(string):
    counter=0
    lst=[]
    for i in range(len(string)):
        if string[i]=='.' and counter<1 :
            lst.append(string[i])
            counter+=1
        elif string[i]!='.':
            lst.append(string[i])
    res="".join(lst)
    return res



import csv
res_list=[]
print(remove_extradots("-0.29444.2"))
with open('A.csv') as file2:
    reader=csv.reader(file2)

    for row in reader:
        data=[]
        for i in range(len(row)):
            
            data.append(remove_extradots(row[i]))
        res_list.append(data)
    


with open('modified_A.csv', 'w') as file3:
    writer = csv.writer(file3)
    writer.writerows(res_list)
file3.close()
                


# %%
#test file A
import pandas as pd
df_A=pd.read_csv("A.csv")
df_A.shape
df_A2=pd.read_csv("A_modified.csv")
df_A2.shape


# %%
#a block that is used to concate the drug names(drug combinations and related concentration)

import csv
res_list=[]
with open('train.csv', 'r') as file:
    reader = csv.reader(file)
    
    for row in reader:
        data=[]
        data.append(row[0]+'+'+row[1]+"+"+row[2])
        res_list.append(data)

        
        
with open('concatenated.csv', 'w') as file2:
    writer = csv.writer(file2)
    writer.writerows(res_list)

# %%
#a block that is used to concate concentration and ks numbers.
import pandas as pd
f1 = pd.read_csv('concatenated.csv')
#f2 = pd.read_csv('C.csv')
f3=pd.read_csv('D.csv')
f4=pd.read_csv('E.csv')
f5=pd.read_csv('modified_A.csv')
file = [f1,f5]
train = pd.concat(file,axis=1)
train.to_csv("train3.csv", index=0, sep=',')

# %%
#a block that sets the fields for the drug data
import csv
fields=['drug_name','default']
for i in range(1,239):
    fields.append('ks'+str(i))
rows=[]
with open('train3.csv', 'r') as file:
    reader = csv.reader(file)
    
    for row in reader:
        rows.append(row)

        
        
with open('drug_withfields2.csv', 'w') as file2:
    writer = csv.writer(file2)
    writer.writerow(fields)
    writer.writerows(rows)

# %%
#do some further processing with "drug_withfields" because there are still some data error in it
def remove_extradots(string):
    counter=0
    lst=[]
    for i in range(len(string)):
        if string[i]=='.' and counter<1 :
            lst.append(string[i])
            counter+=1
        elif string[i]!='.':
            lst.append(string[i])
    res="".join(lst)
    return res



import csv
res_list=[]
print(remove_extradots("-0.29444.2"))
with open('drug_withfields2.csv') as file2:
    reader=csv.reader(file2)

    for row in reader:
        data=[]
        for i in range(len(row)):
            
            data.append(remove_extradots(row[i]))
        res_list.append(data)
    


with open('drug_with_fields.csv', 'w') as file3:
    writer = csv.writer(file3)
    writer.writerows(res_list)
file3.close()
                

# %%
#we take etoposide and palitaxel out. 
df=pd.read_csv('drug_with_fields.csv')
df11=df.loc[(df['drug_name'] =='Etoposide+Etoposide+1') ]
df12=df.loc[(df['drug_name'] =='Etoposide+Etoposide+2') ]
df13=df.loc[(df['drug_name'] =='Etoposide+Etoposide+3') ]
df14=df.loc[(df['drug_name'] =='Etoposide+Etoposide+4') ]
df15=df.loc[(df['drug_name'] =='Etoposide+Etoposide+5') ]
df16=df.loc[(df['drug_name'] =='Etoposide+Etoposide+6') ]
df21=df.loc[(df['drug_name'] =='Paclitaxel+Paclitaxel+1') ]
df22=df.loc[(df['drug_name'] =='Paclitaxel+Paclitaxel+2') ]
df23=df.loc[(df['drug_name'] =='Paclitaxel+Paclitaxel+3') ]
df24=df.loc[(df['drug_name'] =='Paclitaxel+Paclitaxel+4') ]
df25=df.loc[(df['drug_name'] =='Paclitaxel+Paclitaxel+5') ]
df26=df.loc[(df['drug_name'] =='Paclitaxel+Paclitaxel+6') ]
df31=df.loc[(df['drug_name'] =='Etoposide+Paclitaxel+1')]
df32=df.loc[(df['drug_name'] =='Etoposide+Paclitaxel+2')]
df33=df.loc[(df['drug_name'] =='Etoposide+Paclitaxel+3')]
df34=df.loc[(df['drug_name'] =='Etoposide+Paclitaxel+4')]
df35=df.loc[(df['drug_name'] =='Etoposide+Paclitaxel+5')]
df36=df.loc[(df['drug_name'] =='Etoposide+Paclitaxel+6')]
df41=df.loc[(df['drug_name'] =='Paclitaxel+Etoposide+1') ]
df42=df.loc[(df['drug_name'] =='Paclitaxel+Etoposide+2') ]
df43=df.loc[(df['drug_name'] =='Paclitaxel+Etoposide+3') ]
df44=df.loc[(df['drug_name'] =='Paclitaxel+Etoposide+4') ]
df45=df.loc[(df['drug_name'] =='Paclitaxel+Etoposide+5') ]
df46=df.loc[(df['drug_name'] =='Paclitaxel+Etoposide+6') ]

frames = [df11, df12,df13,df14,df15,df16,df21,df22,df23,df24,df25,df26,df31,df32,df33,df34,df35,df36,df41,df42,df43,df44,df45,df46]

e_and_p = pd.concat(frames)
#etoposide and paclitaxel combination. 
e_and_p

# %%
#a code block that is used to deal with the data using PCA models(dimensionality reduction)
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

X_drug = e_and_p.iloc[:, 3:240].values

X_drug = StandardScaler().fit_transform(X_drug)

pca_drug = PCA(n_components=2)
components = pca_drug.fit_transform(X_drug)
principalDf = pd.DataFrame(data = components
             , columns = ['principal component 1', 'principal component 2'])
principalDf


# %%
#compute the distance between points
#calculating distances between centroids
#index for reference
#0，6，12，18
#1，7，13，19
#2,8,14,20
#3,9,15,21
#4,10,16,22
#5，11，17，23
#first, we gonna find the most similar single drug response for "Etoposide+Paclitaxel+1	"
import math
p_combi=principalDf.iloc[12,:]
dist1=[]

p=[]
for i in range(24):
    p.append([principalDf.iloc[i,:][0],principalDf.iloc[i,:][1]])
dist1.append(math.dist(p[12],p[0]))

dist1.append(math.dist(p[12],p[6]))
#print(dist1)
#conclusion, the most similar single drug response for "Etoposide+Paclitaxel+1	" is 	"Etoposide+Etoposide+1"

#then, we see the situation where the concentration decreases(still etoposide +paclitaxel)
dist2=[]
dist2.append(math.dist(p[13],p[1]))
dist2.append(math.dist(p[13],p[7]))
#print(dist2)
#conclusion: still "etoposide+etoposide+1"

#concentration 3
dist3=[]
dist3.append(math.dist(p[14],p[2]))
dist3.append(math.dist(p[14],p[8]))
dist3

#then we just write a loop function and see the whole 6 situations
big_dist=[]
for i in range(6):
    dist=[]
    dist.append(math.dist(p[i+12],p[i]))
    dist.append(math.dist(p[i+12],p[i+6]))
    big_dist.append(dist)
print(big_dist)
counter_for_etoposide=0
counter_for_paclitaxel=0
for i in range(len(big_dist)):
    if big_dist[i][0]<big_dist[i][1]:
        counter_for_etoposide+=1
    else:
        counter_for_paclitaxel+=1
print(counter_for_etoposide,counter_for_paclitaxel)


#p1=principalDf.iloc[0,:]
#p2=principalDf.iloc[6,:]

#conclusion: etoposide is the major drug in "etoposide+paclitaxel" combination



# %%
#test a datatype for debugging
type(principalDf.iloc[0,:][0])

# %%
#find a similar single drug response for "paclitaxel+etoposide" combination

big_dist2=[]
for i in range(6):
    dist=[]
    dist.append(math.dist(p[i+18],p[i]))
    dist.append(math.dist(p[i+18],p[i+6]))
    big_dist2.append(dist)
print(big_dist2)
counter_for_etoposide2=0
counter_for_paclitaxel2=0
for i in range(len(big_dist2)):
    if big_dist2[i][0]<big_dist2[i][1]:
        counter_for_etoposide2+=1
    else:
        counter_for_paclitaxel2+=1
print(counter_for_etoposide2,counter_for_paclitaxel2)

# %%
#processing with the data using centroid
#did not use centroid before because of different concentration
#centroid_for_e=
import math
p_for_e=principalDf.iloc[0:6,:]
p_for_p=principalDf.iloc[6:12,:]
p_for_eandp=principalDf.iloc[12:18,:]
p_for_pande=principalDf.iloc[18:24,:]
centroid1=p_for_e.mean(axis=0)
centroid2=p_for_p.mean(axis=0)
centroid3=p_for_eandp.mean(axis=0)
centroid4=p_for_pande.mean(axis=0)

distance=[]
distance2=[]
distance.append(math.dist(centroid3,centroid1))
distance.append(math.dist(centroid3,centroid2))
distance2.append(math.dist(centroid4,centroid1))
distance2.append(math.dist(centroid4,centroid2))
print(distance)
print(distance2)

#for "etoposide+paclitaxel", etoposide is more similar to it
#for "paclitaxel+etoposide", paclitaxel is more similar to it


# %%

#sklearn codes about knn
from sklearn.neighbors import NearestNeighbors
neighbor = NearestNeighbors(n_neighbors=2)
neighbor.fit(p)

print(neighbor.kneighbors([p[12]]))



# %%
#before using knn, we need to process with p(add a label. drug_name. for further classification)
#for i in range(6):
    #print(p[i])

#e_c=pd.Series(["Etoposide"], dtype="string")
#e_p=pd.Series(["Paclitaxel"], dtype="string")
#e_cp=pd.Series(["Etoposide+Paclitaxel"], dtype="string")
#type(e_c)
#newdata=[]
#for i in range(6):

#    newdata.append(pd.concat([p[i],e_c]))
#for j in range(6):
#    newdata.append(pd.concat([p[i+6],e_p]))
#for k in range(6):
 #   newdata.append(pd.concat([p[i+12],e_cp]))

for i in range(6):
    p[i].append('Etoposide')
for j in range(6):
    p[j+6].append('Paclitaxel')
for k in range(6):
    p[k+12].append('Etoposide+Paclitaxel')
for m in range(6):
    p[m+18].append("Paclitaxel+Etoposide")





# %%
#to show p

# %%
p2=p[:12]

p2

# %%
def get_centroid(point_lst):
    x_sum=0
    y_sum=0
    for i in range(len(point_lst)):
        x_sum+=point_lst[i][0]
        y_sum+=point_lst[i][1]

    x_mean=x_sum/len(point_lst)
    y_mean=y_sum/len(point_lst)
    centroid=[]
    centroid.append(x_mean)
    centroid.append(y_mean)
    centroid.append(point_lst[0][-1])
    return centroid
print(get_centroid(p[12:18]))


# %%
#hand-written codes about knn
#1 euclidean distance
# Example of calculating Euclidean distance
from math import sqrt
 
# calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
	distance = 0.0
	for i in range(len(row1)-1):
		distance += (row1[i] - row2[i])**2
	return sqrt(distance)
# Locate the most similar neighbors
def get_neighbors(train, test_row, num_neighbors):
	distances = list()
	for train_row in train:
		dist = euclidean_distance(test_row, train_row)
		distances.append((train_row, dist))
	distances.sort(key=lambda tup: tup[1])
	neighbors = list()
	for i in range(num_neighbors):
		neighbors.append(distances[i][0])
	return neighbors

# Make a classification prediction with neighbors
def predict_classification(train, test_row, num_neighbors):
    	
	neighbors = get_neighbors(train, test_row, num_neighbors)
	output_values = [row[-1] for row in neighbors]
	prediction = max(set(output_values), key=output_values.count)
	return prediction
 
# Test distance function

row0 = p[12]
for row in p:
	distance = euclidean_distance(row0, row)
	#print(distance)
neighbors = get_neighbors(p2, p[12], 3)
#for neighbor in neighbors:
	#print(neighbor)
#neighbors2=get_neighbors(newdata,newdata[12],3)
#prediction = predict_classification(p2, p[12], 3)
#print('The single drug response that is similar to " %s " is: %s.' % (p[12][-1], prediction))
#prediction = predict_classification(p2, p[12], 4)
#print('The single drug response that is similar to " %s " is: %s.' % (p[12][-1], prediction))
#prediction = predict_classification(p2, p[12], 5)
#print('The single drug response that is similar to " %s " is: %s.' % (p[12][-1], prediction))
#prediction = predict_classification(p2, p[12], 6)
#print('The single drug response that is similar to " %s " is: %s.' % (p[12][-1], prediction))
#prediction = predict_classification(p2, p[12], 7)
#print('The single drug response that is similar to " %s " is: %s.' % (p[12][-1], prediction))
centroid_ep=get_centroid(p[12:18])
#prediction = predict_classification(p2, centroid_ep, 3)
#print('The single drug response that is similar to " %s " is: %s.' % (centroid_ep[-1], prediction))
centroid_pe=get_centroid(p[18:24])
new_lst=[]
new_lst.append(centroid_pe)
new_lst.append(centroid_ep)
new_centroid=get_centroid(new_lst)
#prediction = predict_classification(p2, centroid_pe, 3)
#print('The single drug response that is similar to " %s " is: %s.' % (centroid_pe[-1], prediction))
print(new_centroid)
res=[]
for i in range(1,13):
    res.append(predict_classification(p2, new_centroid, i))
prediction = predict_classification(p2, new_centroid, 12)
print('The single drug response that is similar to " %s " is: %s.' % (new_centroid[-1], prediction))
print(res)



# %%
#find similar drug response


# %%
#visualization(not the main work; just for getting an intuition)
#pca model
"""
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


X_drug = e_and_p.iloc[:, 2:240].values

X_drug = StandardScaler().fit_transform(X_drug)

pca_drug = PCA(n_components=2)
components = pca_drug.fit_transform(X_drug)



fig = px.scatter(components, x=0, y=1, color=e_and_p['drug_name'])
fig.show()
"""

# %%



