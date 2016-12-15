1 import easygui
2 import warnings
3 warnings.filterwarnings('ignore')
4 import pandas as pd
5 import numpy as np
6 import requests
7 import matplotlib.pyplot as plt
8 # import seaborn as sns
9 from operator import itemgetter
10 from sklearn.datasets import load_boston
11 from sklearn.ensemble.forest import RandomForestRegressor
12 from sklearn.cross_validation import ShuffleSplit
13 from sklearn.metrics import r2_score
14 from sklearn.metrics import mean_squared_error
15 from collections import defaultdict
16 from sklearn import preprocessing
17 import csv
18 from sklearn.ensemble import GradientBoostingRegressor
19 from sklearn import cross_validation
20 from sklearn.neighbors import NearestNeighbors
21
22 print "(-:!Comment!:-) Srart reading data **"
23
24 train = pd.read_csv("C:/Indiana Stuff/Sem 1/Data Mining/DM Project//train.csv")
25 kaggle_test = pd.read_csv("C:/Indiana Stuff/Sem 1/Data Mining/DM Project//test.csv")
26 zip = pd.read_csv("C:/Indiana Stuff/Sem 1/Data Mining/DM Project//neighbourhood_zips.csv")
27
28 print "(-:!Comment!:-) Finished reading data **"
29 print "(-:!Comment!:-) separating out sale price and merging training and testing datasets for cleaning **"
30 test_ids = kaggle_test['Id'].copy()
31 Target = train['SalePrice'].copy()
32 del train['SalePrice']
33 frames = [train, kaggle_test]
34 train_1 = pd.concat(frames, keys=['train', 'kaggle_test'])
35
36 ############################## Replacing Nulls #########################################################
37 print "(-:!Comment!:-) Data Pre Procesing Start **"
38
39 print "(-:!Comment!:-) replacing Nulls in data **"
40 train_1.LotFrontage[train_1.LotFrontage.isnull()] = 0 #replacing nulls in LotFrontage
41 train_1.MasVnrArea[train_1.MasVnrArea.isnull()] = 0 #replacing nulls in MasVnrArea
42 train_1.GarageYrBlt[train_1.GarageYrBlt.isnull()] = 2020 #replacing nulls in GarageYrBlt
43 train_1.MasVnrType[train_1.MasVnrType.isnull()] = 'None' #replacing nulls in MasVnrType
44
45 train_1.Electrical[train_1.Electrical.isnull()] = train_1.Electrical.mode().item() #replacing nulls in Electrical
with mode of data
46
47 for c in train_1:
48 if train_1[c].dtype == object:
49 train_1.loc[train_1[c].isnull(),c] = 'NA' #replacing nulls in all Categorical Variables
50
51 samples = train_1.loc[not train_1[c].isnull(),c]
52 neigh = NearestNeighbors(n_neighbors=1)
53 neigh.fit(samples)
54 NearestNeighbors(algorithm='auto', leaf_size=30, ...)
55 print(neigh.kneighbors([[1., 1., 1.]]))
56
57 for c in train_1:
58 if train_1[c].dtype <> object:
59 train_1.loc[train_1[c].isnull(),c] = 0 #replacing nulls in all Continuous Variables
60
61 print "(-:!Comment!:-) finished replacing Nulls in data **"
62 print "(-:!Comment!:-) Converting neighbourhood to zip code **"
63 train_2 = pd.merge(train_1, zip, left_on='Neighborhood', right_on='Neighbourhood', how = 'left')
64 print "(-:!Comment!:-) Finished Converting neighbourhood to zip code **"
65
66 ############Converting ordinal variables into numeric ###########################
67 print "(-:!Comment!:-) Converting ordinal variables to numeric **"
68
69 train_2['ExterQual'] = train_2['ExterQual'].map({'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1})
70 train_2['ExterCond'] = train_2['ExterCond'].map({'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1})
71 train_2['BsmtQual'] = train_2['BsmtQual'].map({'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'NA':0})
72 train_2['BsmtCond'] = train_2['BsmtCond'].map({'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'NA':0})
73 train_2['BsmtExposure'] = train_2['BsmtExposure'].map({'Gd':5, 'Av':4, 'Mn':3, 'No':1, 'NA':0})
74 train_2['HeatingQC'] = train_2['HeatingQC'].map({'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'NA':0})
75 train_2['KitchenQual'] = train_2['KitchenQual'].map({'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'NA':0})
76 train_2['FireplaceQu'] = train_2['FireplaceQu'].map({'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'NA':0})
77 train_2['GarageQual'] = train_2['GarageQual'].map({'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'NA':0})
78 train_2['GarageCond'] = train_2['GarageCond'].map({'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'NA':0})
79 train_2['PoolQC'] = train_2['PoolQC'].map({'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'NA':0})
80 print "(-:!Comment!:-) Finished onverting ordinal variables to numeric **"
81 train_3 = train_2.copy()
82 print "(-:!Comment!:-) Deleting unnecessary columns **"
83 del train_3['Id']
84 del train_3['Neighborhood']
85 del train_3['MiscFeature']
86 del train_3['Neighbourhood']
87
88 print "(-:!Comment!:-) finished Deleting unnecessary columns **"
89 def convert(column):
90 number = preprocessing.LabelEncoder()
91 return number.fit_transform(column)
92
93 print "(-:!Comment!:-) Converting string to category data type **"
94 number = preprocessing.LabelEncoder()
95
96 for col in train_3:
97 if train_3[col].dtype == object:
98 train_3.loc[:,col] = train_3.loc[:, col].astype('category')
99
100 print "(-:!Comment!:-) Finished Converting string to category data type **"
101
102 print "(-:!Comment!:-) Starting data prep for random forests for feature selection **"
103 print "(-:!Comment!:-) Separating out training and test data for random forest**"
104
105 train_4 = train_3.iloc[0:len(train),:]
106 # test_clean = train_3.iloc[len(train):,:]
107
108 print "(-:!Comment!:-) Converting categories to numbers to pass into random forests **"
109
110 for col in train_4:
111 if str(train_4[col].dtype) == 'category':
112 train_4.loc[:,col] = convert(train_4.loc[:,col])
113
114 print "(-:!Comment!:-) Data Ready for Random Forests, iterating random forests **"
115
116 X = train_4
117 Y = Target
118
119 # # ShuffleSplit(Total number of records, n_iter= number of iterations , test_size = fraction or number of rows to
give in test data, rest of the rows or fraction will go to training data):
120
121 rf = RandomForestRegressor()
122 scores = defaultdict(list)
123 names = list(X.columns.values)
124 # print(X.shape)
125 # print (Y.shape)
126 # crossvalidate the scores on a number of different random splits of the data
127 for (train_idx, test_idx) in ShuffleSplit(len(X),n_iter=100,test_size = 0.2):
128 temp = []
129 X_train, X_test = X.loc[train_idx], X.loc[test_idx]
130 Y_train, Y_test = Y.loc[train_idx], Y.loc[test_idx]
131 r = rf.fit(X_train, Y_train)
132 acc = r2_score(Y_test, rf.predict(X_test))
133
134 for i in range(X.shape[1]):
135 X_t = X_test.copy()
136 X_t.iloc[:,i] = np.random.permutation(X_t.iloc[:,i].tolist())
137 shuff_acc = r2_score(Y_test, rf.predict(X_t))
138 scores[names[i]].append(abs(acc-shuff_acc)/acc)
139 print "(-:!Comment!:-) Finished Random Forests**"
140
141 sorted_scores = sorted([(feat, round(np.mean(score), 4) ) for feat, score in scores.items()], reverse=True, key=it
emgetter(1))
142 sorted_df = pd.DataFrame(columns=('column_name', 'score'))
143 with open('C:/Indiana Stuff/Sem 1/Data Mining/DM Project//scores.csv', 'wb') as csv_file:
144 writer = csv.writer(csv_file)
145 for i in sorted_scores:
146 writer.writerow(i)
147 sorted_df.loc[len(sorted_df)] = i
148
149
150 train_3['Zip'] = train_3['Zip'].astype('category')
151
152 reduced_features = []
153 for i in range(40):
154 reduced_features.append(sorted_scores[i][0])
155
156
157 X_reduced = train_3.loc[:,reduced_features]
158 print "(-:!Comment!:-) Done with Random Forests, we have the top 40 varibles sorted by score of importance **"
159 print "(-:!Comment!:-) Checking multicolleniarity between continuous variables **"
160
161 cont_features = []
162 print X_reduced.shape
163 for col in X_reduced:
164 if str(X_reduced[col].dtype) != 'category':
165 cont_features.append(col)
166
167 X_Cont = X_reduced.loc[:,cont_features]
168
169 X_Corr = X_Cont.corr().abs()
170 X_Corr["column1"] = X_Corr.columns.values
171 X_Corr = pd.melt(X_Corr, id_vars=["column1"], value_vars=cont_features, var_name = "column2",value_name='corr_val'
) 172 X_Corr =
X_Corr[X_Corr['corr_val']>=0.7]
173 X_Corr = X_Corr[X_Corr['corr_val']<1]
174
175
176
177
178 ##Removing highly correlated variables (more than 0.7 correlation)
179 print "(-:!Comment!:-) Removing one of the highly correlated variables **"
180
181 X_Corr = pd.merge(X_Corr, sorted_df, left_on='column1', right_on='column_name', how = 'left')
182 X_Corr = pd.merge(X_Corr, sorted_df, left_on='column2', right_on='column_name', how = 'left')
183 del X_Corr['column_name_x']
184 del X_Corr['column_name_y']
185 X_Corr.to_csv("C:/Indiana Stuff/Sem 1/Data Mining/DM Project//Correlations.csv", sep=',')
186
187 remove_cols = []
188
189 for index, row in X_Corr.iterrows():
190 if row['score_x'] < row['score_y']:
191 remove_cols.append(row['column1'])
192 else:
193 remove_cols.append(row['column2'])
194
195 remove_cols = np.unique(remove_cols)
196 for i in remove_cols:
197 del X_reduced[i]
198 print "removed " + i
199
200
201 # Converting categorical variables to numeric
202 print "(-:!Comment!:-) Converting categorical variables to numeric dummies **"
203
204 X_reduced= pd.get_dummies(X_reduced)
205 X_reduced.to_csv("C:/Indiana Stuff/Sem 1/Data Mining/DM Project//train_reduced.csv", sep=',')
206
207 test_clean = X_reduced.iloc[len(train):,:]
208 train_clean = X_reduced.iloc[:len(train),:]
209 test_clean.to_csv("C:/Indiana Stuff/Sem 1/Data Mining/DM Project//test_clean.csv", sep=',')
210 X_reduced.to_csv("C:/Indiana Stuff/Sem 1/Data Mining/DM Project//X_reduced.csv", sep=',')
211
212
213 print "(-:!Comment!:-) Finished Data Pre Processing **"
214
215
216 print "(-:!Comment!:-) Startng to run GradientBoostingRegressor model **"
217
218 #############################Actual Model Building
219 num_folds = 5
220 num_instances = len(train_clean)
221 X = train_clean
222 Y = Target
223
224
225 # for (train_idx, test_idx) in ShuffleSplit(len(train_clean),n_iter=1,test_size = 0.2):
226 # X_train_r, X_test_r = X.loc[train_idx], X.loc[test_idx]
227 # Y_train_r, Y_test_r = Y.loc[train_idx], Y.loc[test_idx]
228
229 kf_total = cross_validation.KFold(num_instances, n_folds=num_folds, shuffle=True, random_state=4)
230 params = {'n_estimators': 50, 'max_depth': 40,'learning_rate': 0.5, 'loss': 'huber'}
231 clf = GradientBoostingRegressor(**params)
232
233 # for train, test in kf_total:
234 # print train, '\n', test, '\n\n'
235 for train_indices, test_indices in kf_total:
236 clf.fit(X.iloc[train_indices,:], Y.iloc[train_indices])
237
238
239 print "Negative Mean Absolute Error"
240 print cross_validation.cross_val_score(clf, X, Y, cv=kf_total, n_jobs = 1, scoring='neg_mean_absolute_error')
241 print "R squared error"
242 print cross_validation.cross_val_score(clf, X, Y, cv=kf_total, n_jobs = 1, scoring='r2')
243 print "(-:!Comment!:-) Finished running GradientBoostingRegressor model **"
244 print "(-:!Comment!:-) Predicting values for test data **"
245
246 test_predicted = pd.DataFrame({'SalePrice': clf.predict(test_clean)})
247 result = pd.concat([test_ids,test_predicted], axis=1)
248
249 result.to_csv("C:/Indiana Stuff/Sem 1/Data Mining/DM Project//kagggle_test_predicted.csv", sep=',',index=False)
250
251 easygui.msgbox("All Done!", title="Kaggle Housing Project")
252
253 print "(-:!Comment!:-) All done **"
254
