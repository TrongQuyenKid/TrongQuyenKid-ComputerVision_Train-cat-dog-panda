import os

# Đường dẫn đến thư mục chứa ảnh
image_dir0 = r"datasetss\animals\cats"
image_dir1 = r"datasetss\animals\dogs"
image_dir2 = r"datasetss\animals\panda"
# Định dạng tệp tin ảnh được chấp nhận
accepted_extensions = [".jpg", ".jpeg", ".png", ".bmp"]

# Đếm số lượng tệp tin ảnh
image_count0 = sum(
    1 for file_name in os.listdir(image_dir0)
    if os.path.isfile(os.path.join(image_dir0, file_name))
    and any(file_name.lower().endswith(ext) for ext in accepted_extensions)
)
image_count1 = sum(
    1 for file_name in os.listdir(image_dir1)
    if os.path.isfile(os.path.join(image_dir1, file_name))
    and any(file_name.lower().endswith(ext) for ext in accepted_extensions)
)
image_count2 = sum(
    1 for file_name in os.listdir(image_dir2)
    if os.path.isfile(os.path.join(image_dir2, file_name))
    and any(file_name.lower().endswith(ext) for ext in accepted_extensions)
)

print(f"Số lượng tệp tin ảnh cats: {image_count0}")
print(f"Số lượng tệp tin ảnh dogs: {image_count1}")
print(f"Số lượng tệp tin ảnh panda: {image_count2}")