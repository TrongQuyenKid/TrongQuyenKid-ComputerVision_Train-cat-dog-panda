# import thư viện cần thiết
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
import numpy as np

vao = 110
nhan = 5

# Nạp ảnh đầu vào và convert vào mảng NumPy rồi thay đổi kích thước, xác định chiều
for i in range(1, vao):
	print("[INFO] Nạp ảnh...")
	link = "input/cats/cats_ (" + str(i) +").jpg"
	print(link)
	image = load_img(link)   # Khai báo đường dẫn chứa ảnh đầu vào

	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)
	# Khởi tạo b sinh ảnh
	aug = ImageDataGenerator(
		rotation_range=30,
		zoom_range=0.15,
		width_shift_range=0.2,
		height_shift_range=0.2,
		shear_range=0.15,
		horizontal_flip=True,
		fill_mode="nearest")
	total = 0

	print("[INFO] Sinh ảnh...")
	imageGen = aug.flow(image, batch_size=1, save_to_dir="output/cats", save_prefix="image", save_format="jpg")

	for image in imageGen:
		# tăng biến đếm
		total += 1
		if total == nhan:  # Tạo 100 ảnh
			break

for i in range(1, vao):
	print("[INFO] Nạp ảnh...")
	link = "input/dogs/dogs_  (" + str(i) +").jpg"
	print(link)
	image = load_img(link)   # Khai báo đường dẫn chứa ảnh đầu vào

	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)
	# Khởi tạo b sinh ảnh
	aug = ImageDataGenerator(
		rotation_range=30,
		zoom_range=0.15,
		width_shift_range=0.2,
		height_shift_range=0.2,
		shear_range=0.15,
		horizontal_flip=True,
		fill_mode="nearest")
	total = 0

	print("[INFO] Sinh ảnh...")
	imageGen = aug.flow(image, batch_size=1, save_to_dir="output/dogs", save_prefix="image", save_format="jpg")

	for image in imageGen:
		# tăng biến đếm
		total += 1
		if total == nhan:  # Tạo 100 ảnh
			break

for i in range(1, vao):
	print("[INFO] Nạp ảnh...")
	link = "input/panda/panda_ (" + str(i) +").jpg"
	print(link)
	image = load_img(link)   # Khai báo đường dẫn chứa ảnh đầu vào

	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)
	# Khởi tạo b sinh ảnh
	aug = ImageDataGenerator(
		rotation_range=30,
		zoom_range=0.15,
		width_shift_range=0.2,
		height_shift_range=0.2,
		shear_range=0.15,
		horizontal_flip=True,
		fill_mode="nearest")
	total = 0

	print("[INFO] Sinh ảnh...")
	imageGen = aug.flow(image, batch_size=1, save_to_dir="output/panda", save_prefix="image", save_format="jpg")

	for image in imageGen:
		# tăng biến đếm
		total += 1
		if total == nhan:  # Tạo 100 ảnh
			break