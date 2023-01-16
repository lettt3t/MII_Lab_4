import csv
import pandas as pds
import numpy as nmp
import pylab as graph
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


def graph_def(k, er, swt, crс, dta, clrs, cls):
    graph.subplot(3, 1, 1)
    graph.plot([i for i in range(1, k + 1)], er)
    graph.title("Error plot for various k")
    graph.xlabel("k")
    graph.ylabel("Error")

    colour_list = [clrs[str(i)] for i in cls]
    graph.subplot(3, 1, 2)
    graph.scatter(swt, crс, c=colour_list)
    graph.title("Input Data chart")
    graph.xlabel("Сладость")
    graph.ylabel("Хруст")

    colour_list = [clrs[str(i)] for i in dta]

    graph.subplot(3, 1, 3)
    graph.scatter(swt, crс, c=colour_list)
    graph.title("Input Data chart")
    graph.xlabel("Сладость")
    graph.ylabel("Хруст")
    # pylab.show()


def method_knn(lData, tData, k, w_size, class_num):
    Datas = []
    for i in range(len(lData)):
        Datas.append(lData[i])
    for j in range(len(tData)):
        Datas.append(tData[j])

    l_size = len(lData) - 1
    t_size = len(Datas) - 1 - l_size

    k_max = k
    final_dist = nmp.zeros((t_size, l_size))

    for i in range(t_size):
        for j in range(l_size):
            final_dist[i][j] = (
                                       (int(Datas[l_size + 1 + i][1]) - int(Datas[j + 1][1])) ** 2
                                       + (int(Datas[l_size + 1 + i][2]) - int(Datas[j + 1][2])) ** 2
                               ) ** (1 / 2)

    er_k = [0] * k_max
    for k in range(k_max):
        print("Classification for k:", k + 1)
        sucsess = 0
        er = [0] * t_size
        class_s = [0] * t_size

        for i in range(t_size):
            qwant_dist = [0] * class_num
            print(str(i) + ". " + "We classify the product ", Datas[l_size + i + 1][0])
            tmp = nmp.array(final_dist[i, :])
            dist_max = max(tmp)

            for j in range(k + 1):
                ind_min = list(tmp).index(min(tmp))
                if tmp[j] < w_size:
                    qwant_dist[int(Datas[ind_min + 1][3])] += dist_max - tmp[j]
                else:
                    qwant_dist[int(Datas[ind_min + 1][3])] += 0

                tmp[ind_min] = 1000
                max1 = max(qwant_dist)

                print("neighbor index = " + str(ind_min) + ", neighbor - " + Datas[ind_min + 1][0])
                print("расстояние " + str(qwant_dist))
            class_ind = list(qwant_dist).index(max1)
            class_s[i] = class_ind

            print("Assigning a class:" + Datas[l_size + i + 1][3])
            print(class_s[i])
            print(Datas[l_size + i + 1][3])
            if int(class_s[i]) == int(Datas[l_size + i + 1][3]):
                print("True")
                sucsess += 1
                er[i] = 0
            else:
                print("False")
                er[i] = 1

        er_k[k] = nmp.mean(er)

        print("Error for " + str(k) + " neighbor")
        print(er_k)

    return er_k, class_s


def method_sklearn(dat, clas, k, tsz):
    X_train, X_test, y_train, y_test = train_test_split(dat, clas, test_size=tsz, random_state=0)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print("Training sample parameters")
    print(X_train)
    print("Test sample options")
    print(X_test)
    print("Training sample class_s")
    print(y_train)
    print("Training sample class_s")
    print(y_test)
    print("Result")
    print(predictions)
    return X_train, X_test, y_train, y_test, predictions


# number 1
print("===============================================================")
print("Solver #1")
print("Input Data for products")
Data = [["Продукт", "Сладость", "Хруст", "Класс"], ["Яблоко", "7", "7", "0"], ["Салат", "2", "5", "1"],
        ["Бекон", "1", "2", "2"], ["Орехи", "1", "5", "2"], ["Рыба", "1", "1", "2"], ["Сыр", "1", "1", "2"],
        ["Бананы", "9", "1", "0"], ["Морковь", "2", "8", "1"], ["Виноград", "8", "1", "0"], ["Апельсин", "6", "1", "0"],
        ["Слива", "7", "0", "0"], ["Сельдерей", "0", "6", "1"], ["Шницель", "0", "1", "2"], ["Мандарин", "7", "1", "0"],
        ["Капуста", "0", "7", "1"], ["Сухарики", "2", "8", "2"], ["Ежевика", "8", "2", "0"], ["Огурец", "1", "9", "1"],
        ["Гренки", "3", "9", "2"], ["Манго", "8", "3", "0"], ]
with open("Data_learn.csv", "w", encoding="utf8") as f:
    writer = csv.writer(f, lineterminator="\r")
    for row in Data:
        writer.writerow(row)
print("Incoming Data")
print(Data)
print("===============================================================")
# number2
print("===============================================================")
print("Solver #2")
print("Obtaining a classifier using the KNN method")
k_max = 5
window = 7
er_k, class_s = method_knn(Data[0:11], Data[11:], k_max, window, 3)
Dataset = pds.read_csv("Data_learn.csv")
start_Data = Dataset[:10]["Класс"]
s1 = pds.Series(class_s)
start_Data = pds.concat([start_Data, s1])
sweet = Dataset["Сладость"]
crunch = Dataset["Хруст"]
colours = {"0": "orange", "1": "blue", "2": "green"}
class_s_info = Dataset["Класс"]
print(start_Data)
print(sweet)

graph.figure()
graph_def(k_max, er_k, sweet, crunch, start_Data, colours, class_s_info)
k_max = 2
my_Dataset = pds.read_csv("Data_learn.csv")
sweetness = my_Dataset["Сладость"]
crunch = my_Dataset["Хруст"]
values = nmp.array(list(zip(sweetness, crunch)), dtype=nmp.float64)
class_s = my_Dataset["Класс"]
test_size = 0.5
X_train, X_test, y_train, y_test, predictions = method_sklearn(values, class_s, k_max, test_size)
colours = {"0": "orange", "1": "blue", "2": "green"}
class_s_info = my_Dataset["Класс"]
start_Data = my_Dataset[:10]["Класс"]
s1 = nmp.concatenate((y_train, y_test), axis=0)
s1 = pds.Series(s1)
predictions = pds.Series(predictions)
start_Data = pds.Series(start_Data)
start_Data = pds.concat([start_Data, predictions])

er = 0
ct = 0

truthclass_s = pds.Series(my_Dataset["Класс"])
testclass_s = pds.concat([pds.Series(my_Dataset[:10]["Класс"]), predictions])

print("Error count:")
for i in testclass_s:
    print(str(i) + " " + str(truthclass_s[ct]))
    if i == truthclass_s[ct]:
        er += 0
    else:
        er += 1
ct += 1
er = er / ct
print(er)
er_k = []
for i in range(1, k_max + 1):
    er_k.append(er)
graph_def(k_max, er_k, sweet, crunch, start_Data, colours, class_s_info)
print("===============================================================")

print("===============================================================")
print("Solver 3")
print("New Data for learn")
new_Data = Data[0:11]
new_Data.append(["Мёд", "30", "0", "3"])
new_Data.append(["Хворост", "28", "2", "3"])
new_Data.append(["Зефир", "29", "1", "3"])
new_Data.append(["Мармелад", "26", "0", "3"])

new_Data = new_Data + Data[11:]
new_Data.append(["Фруктовая палочка", "30", "2", "3"])
new_Data.append(["Маршмеллоу", "29", "1", "3"])

print("New Data")
print(new_Data)

with open("Data_learn.csv", "w", encoding="utf8") as f:
    writer = csv.writer(f, lineterminator="\r")
    for row in new_Data:
        writer.writerow(row)

print("\nKnn method for new Data")

k_max = 5
window = 7
er_k, class_s = method_knn(new_Data[0:15], new_Data[15:], k_max, window, 4)
Dataset = pds.read_csv("Data_learn.csv")
start_Data = Dataset[:14]["Класс"]
s1 = pds.Series(class_s)
start_Data = pds.concat([start_Data, s1])

sweet = Dataset["Сладость"]
crunch = Dataset["Хруст"]
colours = {"0": "orange", "1": "blue", "2": "green", "3": "red"}
class_s_info = Dataset["Класс"]
graph_def(k_max, er_k, sweet, crunch, start_Data, colours, class_s_info)

print("\nSk Knn for new Data")

k_max = 2

my_Dataset = pds.read_csv("Data_learn.csv")
sweetness = my_Dataset["Сладость"]
crunch = my_Dataset["Хруст"]

values = nmp.array(list(zip(sweetness, crunch)), dtype=nmp.float64)
class_s = my_Dataset["Класс"]
test_size = 0.461
X_train, X_test, y_train, y_test, predictions = method_sklearn(values, class_s, k_max, test_size)
colours = {"0": "orange", "1": "blue", "2": "green", "3": "red"}
class_s_info = my_Dataset["Класс"]
start_Data = my_Dataset[:14]["Класс"]
s1 = nmp.concatenate((y_train, y_test), axis=0)
s1 = pds.Series(s1)
predictions = pds.Series(predictions)
start_Data = pds.Series(start_Data)
start_Data = pds.concat([start_Data, predictions])

er = 0
ct = 0

truthclass_s = pds.Series(my_Dataset["Класс"])
testclass_s = pds.concat([pds.Series(my_Dataset[:14]["Класс"]), predictions])

print("Error count")
for i in testclass_s:
    print(str(i) + " " + str(truthclass_s[ct]))
    if i == truthclass_s[ct]:
        er += 0
    else:
        er += 1
ct += 1
er = er / ct
print(er)
er_k = []

for i in range(1, k_max + 1):
    er_k.append(er)

graph_def(k_max, er_k, sweet, crunch, start_Data, colours, class_s_info)
graph.show()
print("===============================================================")