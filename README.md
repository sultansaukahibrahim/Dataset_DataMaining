**Analisis & Penjelasan kode Python pada Dataset**

```python
import pandas as pd
```

Mengimpor pustaka **pandas** untuk memanipulasi dan menganalisis data, khususnya data tabular seperti CSV.

---

```python
datasetHybrid = pd.read_csv('Benign Traffic.csv')
```

Membaca file CSV berisi data trafik jaringan normal ke dalam DataFrame **datasetHybrid**.

---

```python
datasetHybrid
```

Menampilkan isi DataFrame **datasetHybrid**.

---

```python
datasetHybrid2 = pd.read_csv('DDoS ICMP Flood.csv')
datasetHybrid2
```

Membaca dan menampilkan dataset DDoS tipe **ICMP Flood**.

---

```python
datasetHybrid3 = pd.read_csv('DDoS UDP Flood.csv')
datasetHybrid3
```

Membaca dan menampilkan dataset DDoS tipe **UDP Flood**.

---

```python
gabungan = pd.concat([datasetHybrid, datasetHybrid2, datasetHybrid3], ignore_index=True)
```

Menggabungkan ketiga dataset menjadi satu DataFrame **gabungan**, dengan index di-reset.

---

```python
gabungan.columns.values
```

Menampilkan nama-nama kolom dari DataFrame **gabungan**.

---

```python
x = gabungan.iloc[:,7: 76]
x
```

Memilih fitur (X) dari kolom ke-7 sampai ke-75.

---

```python
y = gabungan.iloc[:,83]
y
```

Memilih label (Y) dari kolom ke-83 sebagai target klasifikasi.

---

```python
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.2, random_state = 42)
```

Membagi data menjadi 80% data latih dan 20% data uji dengan acakan yang konsisten (**random_state=42**).

---

```python
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
```

Mengimpor modul pohon keputusan dari **scikit-learn**.

---

```python
model = DecisionTreeClassifier(criterion='entropy', splitter = 'random')
model.fit(x_train,y_train)
```

Membuat dan melatih model Decision Tree:

**criterion='entropy'**: menggunakan information gain.
**splitter='random'**: memilih titik split secara acak.

---

```python
y_pred = model.predict(x_test)
y_pred
```

Melakukan prediksi pada data uji (**x_test**) dan menampilkan hasilnya.

---

```python
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
accuracy
```

Menghitung dan menampilkan akurasi prediksi terhadap data uji.

---

```python
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize = (10, 7))
tree.plot_tree(model, feature_names = x.columns.values, class_names = np.array([ 'Benign Traffic','DDos ICMP Flood','DDoS UDP Flood']), filled = True)
plt.show()
```

Menampilkan visualisasi pohon keputusan:

* Menampilkan fitur dan kelas.
**filled=True**: memberikan warna berdasarkan kelas.

---

```python
import seaborn as lol
from sklearn import metrics
label = np.array([ 'Benign Traffic','DDos ICMP Flood','DDoS UDP Flood'])
```

Impor modul **seaborn** (diberi alias **lol**, biasanya **sns**) dan **metrics** untuk evaluasi. Array **label** berisi nama kelas.

---

```python
import matplotlib.pyplot as plt
conf_matrix = metrics.confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 10))
lol.heatmap(conf_matrix, annot=True, cmap='cividis', xticklabels=label, yticklabels=label)
plt.xlabel('Prediksi')
plt.ylabel('Fakta')
plt.show()
```

Menampilkan confusion matrix dalam bentuk heatmap:

Memudahkan analisis kesalahan klasifikasi.
**xticklabels** dan **yticklabels** menunjukkan kelas.
