from __future__ import absolute_import, division, print_function, unicode_literals

import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns; sns.set(style="ticks", color_codes=True)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from tensorflow import contrib
tfe = contrib.eager

sutun_isim = ['MPG','Silindir','MotorHacmi','Beygir','Ağırlık',
                'Hızlanma', 'ModelYılı', 'Menşei'] 

#veriseti okunuyor
raw_dataset = pd.read_csv("auto-mpg.data", names=sutun_isim,
                      na_values = "?", comment='\t',
                      sep=" ", skipinitialspace=True)

dataset = raw_dataset.copy()
dataset = dataset.dropna() #verisetindeki bilinmeyen kısımlar çıkarılıyor

#menşei sutunu sayısal değer içermediği için menşeilerine göre ayrılıyor
origin = dataset.pop('Menşei')
dataset['Amerika'] = (origin == 1)*1.0
dataset['Avrupa'] = (origin == 2)*1.0
dataset['Japonya'] = (origin == 3)*1.0

#eğitim ve test
egitim_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(egitim_dataset.index)

#bazı sutunların değerleri
#sns.pairplot(egitim_dataset[["MPG", "Silindir", "MotorHacmi", "Ağırlık"]], diag_kind="kde")
#plt.show()

#veri istatistikleri
egitim_stats = egitim_dataset.describe()
egitim_stats.pop("MPG")
egitim_stats = egitim_stats.transpose()

#hedefin başlığı çıkarılıyor
egitim_labels = egitim_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')

#veriler arası fark çok olduğu için normalize ediliyor
def norm(x):
    return (x - egitim_stats['mean']) / egitim_stats['std'] #mean=ortalama std=standartsapma
normal_egitim_data = norm(egitim_dataset)
normal_test_data = norm(test_dataset)

#model
def create_model():
    model = keras.Sequential([
            layers.Dense(128, activation=tf.nn.relu, input_shape=[len(egitim_dataset.keys())]),
            layers.Dense(64, activation=tf.nn.relu),
            keras.layers.Dropout(0.5),
            layers.Dense(64, activation=tf.nn.softmax),
            keras.layers.Dropout(0.5),
            layers.Dense(64, activation=tf.nn.relu),
            layers.Dense(1)
        ])
    return model

model = create_model()

optimizer = tf.keras.optimizers.Adam(lr=0.001)

model.compile(loss='mean_squared_error',
            optimizer=optimizer,
            metrics=['mean_absolute_error', 'mean_squared_error', 'accuracy'])


#model inceleme
#print(model.summary())

class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.', end='')

EPOCHS = 1000

#hata değerinin kötüleştiği zaman eğitimi durdurur
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

egitim_surec = model.fit(
    normal_egitim_data, egitim_labels,
    epochs=EPOCHS, validation_split = 0.2,
    callbacks=[early_stop, PrintDot()])

#eğitim sürecini görselleştirme
hist = pd.DataFrame(egitim_surec.history)
hist['epoch'] = egitim_surec.epoch
print(hist.tail())

def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
  
    plt.figure()
    plt.xlabel('Döngü')
    plt.ylabel('Mean Abs Hata [MPG]')
    plt.plot(hist['epoch'], hist['mean_absolute_error'],
           label='Eğitim Hata')
    plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
           label = 'Val Hata')
    plt.ylim([0,5])
    plt.legend()
  
    plt.figure()
    plt.xlabel('Döngü')
    plt.ylabel('Mean Square Hata [$MPG^2$]')
    plt.plot(hist['epoch'], hist['mean_squared_error'],
           label='Eğitim Hata')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'],
           label = 'Val Hata')
    plt.ylim([0,25])
    plt.legend()
    plt.show()

plot_history(egitim_surec)

#test seti tahmin
test_tahmin = model.predict(normal_test_data).flatten()

plt.scatter(test_labels, test_tahmin)
plt.xlabel('Gerçek Değerler [MPG]')
plt.ylabel('Tahminler [MPG]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])
plt.show()

#hata dağılımı
hata = test_tahmin - test_labels
plt.hist(hata, bins = 25)
plt.xlabel("Tahmin Hatası [MPG]")
_ = plt.ylabel("Sayı")
plt.show()