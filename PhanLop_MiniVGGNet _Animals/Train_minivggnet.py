# Import các thư viện
from preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from preprocessing.simplepreprocessor import SimplePreprocessor
from datasetss.simpledatasetloader import SimpleDatasetLoader
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from conv.minivggnet import MiniVGGNet
from imutils import paths
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np

# Chuẩn bị dữ liệu
sp = SimplePreprocessor(32, 32)  # Kích thước ảnh đầu vào: 64x64
iap = ImageToArrayPreprocessor()

print("[INFO] Đang nạp ảnh...")
imagePaths = list(paths.list_images("datasetss/animals"))
sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.astype("float") / 255.0

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

# Khởi tạo danh sách nhãn
label_names = ["cat", "dog", "panda"]

# Tạo ra các biến thể của dữ liệu trong quá trình huấn luyện (Data Augmentation)
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                         horizontal_flip=True, fill_mode="nearest")

# Xây dựng mô hình
print("[INFO] Đang biên dịch model...")
optimizer = SGD(learning_rate=0.01, decay=0.01/60, nesterov=True, momentum=0.9)  
#optimizer = Adam (learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7)
# optimizer = RMSprop(learning_rate=0.001, rho=0.9, momentum=0.0, epsilon=1e-7, centered=False)

model = MiniVGGNet.build(width=32, height=32, depth=3, classes=3)
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

# Train mô hình với Data Augmentation
print("[INFO] Đang training...")
H = model.fit(aug.flow(trainX, trainY, batch_size=32), validation_data=(testX, testY),
              steps_per_epoch=len(trainX) // 32, epochs=60, verbose=1)

# Lưu model
model.save("miniVGGNet.hdf5")

# Đánh giá mô hình
print("[INFO] Đang đánh giá model...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=label_names))

# Vẽ biểu đồ
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 60), H.history["loss"], label="Mất mát khi training")
plt.plot(np.arange(0, 60), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 60), H.history["accuracy"], label="Độ chính xác khi training")
plt.plot(np.arange(0, 60), H.history["val_accuracy"], label="val_acc")
plt.title("Giá trị Loss và độ chính xác trên tập animals")
plt.xlabel("Epoch #")
plt.ylabel("Mất mát/Độ chính xác")
plt.legend()
plt.show()