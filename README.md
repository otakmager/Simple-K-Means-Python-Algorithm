# Simple K-Means Python Algorithm
###### Last Updated: 24 Oktober 2022
#
#
K-Means algorithm adalah salah satu metode dalam unsupervised learning dalam machine learning. Algoritma ini membagi data menjadi beberapa cluster berdasarkan kedekatan kemiripan tanpa diketahui label data.

## Konsep Dasar K-Means Clustering 🔑
- Mengelompokkan 𝑛 objek ke dalam 𝐾 cluster berdasarkan nilai atribut dari
objek tersebut.
- 𝐾 adalah jumlah cluster, berupa bilangan integer positif.
- Merupakan jenis hard clustering → 1 objek hanya dapat menjadi anggota
dari 1 cluster secara eksklusif.
- Membagi data dengan one-level partitioning.
- Memisalkan titik pusat setiap cluster (centroid) untuk perhitungan awal.
- Centroid tidak harus dari actual data.

# The Basic K-means Algorithm 🛠
1.  Select k point as initial centorids
2.  Repeat:
    Form k clusters by assigning each point to its closest centroid
    Recompute the centroid of each cluster
    Until Centroids do not change

## Reminder 📢
1. Project ini dibuat dengan bahasa pemrograman python lewat Google Colab (https://colab.research.google.com/).
2. Dalam project ini menggunakan perhitungan jarak dengan Euclidean distance.
3. Dalam project ini tidak terdapat normalisasi jika range data terlalu besar.

## Import Library 📚
Project ini dijalankan dengan meng-import beberapa library, diantaranya sebagai berikut.

```sh
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
```

## Euclidean Function 📏
Euclidean function digunakan untuk menghitung jarak antar 2 titik. 

```sh
def euclidean(x, p):
  return distance.euclidean(x, p)
```

## K-Means Function 🔍
Digunakan untuk membagi data menjadi k cluster. Dalam function ini nanti akan mencetak detail dari setiap iterasi/langkah.

```sh
def kmeans(cluster, x, y, px, py, itr):
  n = len(x)
  d = np.zeros([cluster, n])
  g = np.zeros([cluster, n])
  nsum = np.zeros(cluster)
  xsum = np.zeros(cluster)
  ysum = np.zeros(cluster)
  compareMin = np.zeros(cluster)
  pxnew = np.zeros(cluster)
  pynew = np.zeros(cluster)

  for i in range(0, n):
    # Mencari jarak titik dengan centroid
    xy = np.array([x[i], y[i]])
    for j in range(0, cluster):
      cent = np.array([px[j], py[j]])
      d[j][i] = euclidean(xy, cent)
      compareMin[j] = d[j][i]
      
    # Clustering berdasarkan nilai paling dekat dengan 0
    idxMin = 0
    minValue = compareMin[0]
    for k in range(0, cluster):
      if ( minValue > compareMin[k] ):
        idxMin = k
        minValue = compareMin[k]
    g[idxMin][i] = 1

  # Membuat nilai centroid baru
  for i in range(0, cluster):
    for j in range(0, n):
      if (g[i][j] == 1):
        nsum[i] = nsum[i] + 1
        xsum[i] = xsum[i] + x[j]
        ysum[i] = ysum[i] + y[j]
  for i in range(0, cluster):
    xmean = xsum[i]/nsum[i]
    ymean = ysum[i]/nsum[i]
    pxnew[i] = xmean
    pynew[i] = ymean

  # Print Detail Komputasi
  print("iterasi ke-", itr)
  print('x = ', x)
  print('y = ', y)
  for i in range(0, cluster):
    cent = np.array([px[i], py[i]])
    print("P ke-", i+1, ': ', cent)
  print("\n")
  
  print("Hasil distance: ")
  print(d)
  print("\n")

  print("Hasil clustering: ")
  print(g)
  print("\n")

  for i in range(0, cluster):
    print("nsum cluster ke-", i, ': ', nsum[i])
    print("xsum cluster ke-", i, ': ', xsum[i])
    print("ysum cluster ke-", i, ': ', ysum[i])
    print('\n')

  print('pnew:')
  for i in range(0, cluster):
    cent = np.array([pxnew[i], pynew[i]])
    print("P new ke-", i+1, ': ', cent)
  print("\n==============================================================\n")

  # Cek centroid awal dan centroid baru apakah sama
  loop = 0
  valid = True
  while(valid and loop < cluster):
    if((px[loop] != pxnew[loop]) or (py[loop] != pynew[loop])):
      valid = False
    loop = loop + 1
  
  # Jika nilai centroid baru dan lama berbeda maka lakukan rekursif
  if(valid == False):
    itr = itr + 1
    kmeans(cluster, x, y, pxnew, pynew, itr)
```
##### Sebagai catatan:
- D adalah matriks jarak dari setiap data dengan setiap centroids.
- G adalah matriks clustering, jika bernilai 1 maka termasuk termasuk cluster X.
- P adalah titik centroid.
- Data maupun centroid dipisah nilai x dan y untuk mempermudah komputasi.

## Contoh Kasus
![image](https://user-images.githubusercontent.com/97395139/197446298-b1d00c7d-8c85-4a76-8f82-67246d9db782.png)

## Contoh untuk 2 cluster 🏃‍♂️
Persiapkan data ke dalam bentuk variabel berikut:
```sh
cluster = 2
itr = 0
x = np.array([0.40, 0.22, 0.35, 0.26, 0.08, 0.45])
y = np.array([0.53, 0.38, 0.32, 0.19, 0.41, 0.30])
px = np.array([0.4, 0.2])
py = np.array([0.3, 0.3])
```
Untuk eksekusi:
```sh
kmeans(cluster, x, y, px, py, itr)
```

## Contoh untuk 3 cluster 🏃‍♂️
Persiapkan data ke dalam bentuk variabel berikut:
```sh
cluster = 3
itr = 0
x = np.array([0.40, 0.22, 0.35, 0.26, 0.08, 0.45])
y = np.array([0.53, 0.38, 0.32, 0.19, 0.41, 0.30])
px = np.array([0.4, 0.2, 0.3])
py = np.array([0.3, 0.3, 0.1])
```
Untuk eksekusi:
```sh
kmeans(cluster, x, y, px, py, itr)
```
