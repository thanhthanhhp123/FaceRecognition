import os
import collections
import numpy as np
from PIL import Image
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))
def knn(X_train, y_train, X_test, k=3):
    y_pred = []
    for i in range(len(X_test)):
        distances = [euclidean_distance(X_test[i], x) for x in X_train]
        k_idx = np.argsort(distances)[:k]
        k_labels = [y_train[idx] for idx in k_idx]  
        most_common = collections.Counter(k_labels).most_common(1)
        y_pred.append(most_common[0][0])
    return np.array(y_pred)

def split_train_test(X, y, test_ratio=0.2, random_state=None):
    """
    Tách tập dữ liệu thành tập train và test.
    Parameters
    ----------
    X : numpy.ndarray
        Mảng chứa dữ liệu đầu vào.
    y : numpy.ndarray
        Mảng chứa nhãn đầu ra.
    test_ratio : float, optional
        Tỷ lệ dữ liệu được tách ra làm tập test, mặc định là 0.2.
    random_state : int, optional
        Seed để lấy mẫu ngẫu nhiên, mặc định là None.

    Returns
    -------
    X_train : numpy.ndarray
        Mảng chứa dữ liệu huấn luyện.
    y_train : numpy.ndarray
        Mảng chứa nhãn huấn luyện.
    X_test : numpy.ndarray
        Mảng chứa dữ liệu kiểm tra.
    y_test : numpy.ndarray
        Mảng chứa nhãn kiểm tra.
    """
    # Tính số lượng dữ liệu trong tập test
    n_test = int(len(X) * test_ratio)
    # Thiết lập seed cho hàm np.random.choice
    np.random.seed(random_state)
    # Lấy ngẫu nhiên các chỉ mục để tạo tập test
    test_indices = np.random.choice(len(X), size=n_test, replace=False)
    # Tạo tập train bằng các chỉ mục không có trong tập test
    train_indices = np.array(list(set(range(len(X))) - set(test_indices)))
    # Tách dữ liệu thành tập train và test
    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]
    return X_train, y_train, X_test, y_test


# Đường dẫn tới thư mục chứa dataset
data_dir = r'C:\Users\QuanGutsBoiz\Documents\FaceRecognition\Datasets'

# Danh sách các thư mục con trong dataset, mỗi thư mục tương ứng với một người
person_dirs = [os.path.join(data_dir, d) for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

# Tạo danh sách tên các ảnh và nhãn tương ứng
X = []
y = []

for i, person_dir in enumerate(person_dirs):
    for filename in os.listdir(person_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            # Đọc ảnh và chuyển sang dạng grayscale
            img = Image.open(os.path.join(person_dir, filename)).convert('L')
            # Chuyển ảnh sang mảng numpy
            img = img.resize((100, 100))
            img_np = np.array(img)

            # Thêm ảnh vào danh sách X
            X.append(img_np)
            # Thêm nhãn vào danh sách y
            y.append(i)
            
# Chuyển danh sách X và y sang mảng numpy
X = np.array(X)
y = np.array(y)
X_train,y_train,X_test,y_test=split_train_test(X,y,test_ratio=0.2,random_state=42)
#X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=None)
# In kích thước của X và y
print(f'Shape of X: {X.shape}')
print(f'Shape of y: {y.shape}')
y_pre=knn(X_train, y_train,X_test,k=3)
print(y_pre)
print(y_test)
