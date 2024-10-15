import h5py

def print_attrs(name, obj):
    """Hàm để in ra thuộc tính của đối tượng HDF5."""
    print(f"{name}: {obj}")
    if isinstance(obj, h5py.Group):
        for key, value in obj.attrs.items():
            print(f"  Attribute: {key} - Value: {value}")

def display_h5_info(file_path):
    """Hàm để hiển thị toàn bộ thông tin trong file HDF5."""
    with h5py.File(file_path, 'r') as h5_file:
        print("Danh sách nhóm và dataset trong file:")
        h5_file.visititems(print_attrs)
        print("\nCác thuộc tính của từng nhóm/dataset:")
        for name in h5_file:
            print(f"{name}: {h5_file[name]}")
            print("  Attributes:", h5_file[name].attrs)

# Đường dẫn đến file HDF5 của bạn
file_path = 'rbm.weights.h5'
display_h5_info(file_path)
