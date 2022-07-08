# tugas-uas

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from jcopml.pipeline import num_pipe, cat_pipe
df = pd.read_csv("data/dataset1.csv")
df
jenis_kelamin	pekerjaan	status	ipk	lulus
0	laki	mhs	belum	3.17	1
1	laki	bekerja	belum	3.30	1
2	perempuan	mhs	belum	3.01	1
3	perempuan	mhs	menikah	3.25	1
4	laki	bekerja	menikah	3.20	0
5	laki	bekerja	menikah	2.50	0
6	perempuan	bekerja	menikah	3.00	0
7	perempuan	bekerja	belum	2.70	0
8	laki	bekerja	belum	2.40	0
9	perempuan	mhs	menikah	2.50	0
10	perempuan	mhs	belum	2.50	0
11	perempuan	mhs	belum	3.50	1
12	laki	bekerja	menikah	3.30	1
13	laki	mhs	menikah	3.25	1
14	laki	mhs	belum	2.30	0
X = df.drop(columns="lulus")
y = df.lulus
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
((12, 4), (3, 4), (12,), (3,))
preprocessor = ColumnTransformer([
    ('numeric', num_pipe(), ["ipk"]),
    ('categoric', cat_pipe(encoder='onehot'), ['jenis_kelamin', 'pekerjaan', 'status']),
])
from sklearn.naive_bayes import GaussianNB
pipeline = Pipeline([
    ('prep', preprocessor),
    ('algo', GaussianNB())
])
pipeline.fit(X_train, y_train)
Pipeline(steps=[('prep',
                 ColumnTransformer(transformers=[('numeric',
                                                  Pipeline(steps=[('imputer',
                                                                   SimpleImputer(strategy='median'))]),
                                                  ['ipk']),
                                                 ('categoric',
                                                  Pipeline(steps=[('imputer',
                                                                   SimpleImputer(strategy='most_frequent')),
                                                                  ('onehot',
                                                                   OneHotEncoder(handle_unknown='ignore'))]),
                                                  ['jenis_kelamin', 'pekerjaan',
                                                   'status'])])),
                ('algo', GaussianNB())])
pipeline.score(X_train, y_train)
0.9166666666666666
pipeline.score(X_test, y_test)
1.0
from jcopml.plot import plot_confusion_matrix
plot_confusion_matrix(X_train, y_train, X_test, y_test, pipeline)

Prediction
X_pred = pd.read_csv("data/testing.csv")
X_pred
jenis_kelamin	pekerjaan	status	ipk
0	laki	mhs	belum	2.7
pipeline.predict(X_pred)
array([0], dtype=int64)
X_pred["lulus"] = pipeline.predict(X_pred)
X_pred
jenis_kelamin	pekerjaan	status	ipk	lulus
0	laki	mhs	belum	2.7	0
 
 
