# Tính toán prior, mean và variance cho mỗi lớp
for i in range(num_classes):
    # Lấy các ảnh trong lớp i
    X_class = X_train[y_train == i]
    # Tính toán prior
    prior[i] = X_class.shape[0] / X_train.shape[0]
    # Tính toán mean và variance của từng pixel
    for j in range(num_pixels):
        mean[i,j] = np.mean(X_class[:,j])
        variance[i,j] = np.var(X_class[:,j])

#return prior, mean, variance

def gaussian_pdf(x, mu, sigma):
    """
    Gaussian Probability Density Function
    """
    if sigma == 0:
        if x == mu:
            return 1
        else:
            return 0
    else:
        coeff = 1.0 / (sigma * math.sqrt(2.0 * math.pi))
        exponent = math.exp(-(float(x) - float(mu))**2 / (2.0 * sigma**2))
        return coeff * exponent
def predict_navie_bayes(img, prior, mu, var):
    """
    Dự đoán nhãn của ảnh đầu vào bằng phương pháp Navie Bayes.
    
    Arguments:
    img -- ảnh đầu vào cần dự đoán, có kích thước (100, 100)
    prior -- vector chứa xác suất của mỗi lớp, có kích thước (n_classes,)
    mu -- mảng 2 chiều chứa giá trị trung bình của từng pixel của mỗi lớp, có kích thước (n_classes, 10000)
    var -- mảng 2 chiều chứa phương sai của từng pixel của mỗi lớp, có kích thước (n_classes, 10000)
    
    Returns:
    predicted_label -- nhãn dự đoán cho ảnh đầu vào
    """
    
    # Số lớp
    n_classes = len(prior)
    
    # Tính toán xác suất cho từng lớp
    posterior = np.zeros(n_classes)
    for i in range(n_classes):
        posterior[i] = prior[i]
        for j in range(img.size):
            posterior[i] *= gaussian_pdf(img[j], mu[i][j], var[i][j])
    
    # Chọn lớp có xác suất lớn nhất là nhãn dự đoán cho ảnh đầu vào
    predicted_label = np.argmax(posterior)
    
    return predicted_label
def train_naive_bayes(X_train, y_train, num_classes, num_pixels):
    """
    Huấn luyện mô hình Naive Bayes cho bài toán nhận diện khuôn mặt
    
    Arguments:
    X_train -- ma trận numpy chứa dữ liệu huấn luyện, có kích thước (m, n), m là số mẫu và n là số pixels trong một ảnh
    y_train -- mảng numpy chứa nhãn của dữ liệu huấn luyện, có kích thước (m,)
    num_classes -- số lớp (số người) trong dữ liệu
    num_pixels -- số pixel trong một ảnh
    
    Returns:
    prior -- mảng numpy chứa xác suất tiên nghiệm của mỗi lớp, có kích thước (num_classes,)
    likelihood -- mảng numpy chứa xác suất hậu nghiệm của mỗi pixel với mỗi lớp, có kích thước (num_classes, num_pixels)
    """
    
    # Khởi tạo mảng prior và likelihood
    prior = np.zeros(num_classes)
    likelihood = np.zeros((num_classes, num_pixels))
    
    # Tính xác suất tiên nghiệm cho từng lớp
    for c in range(num_classes):
        prior[c] = np.sum(y_train == c) / y_train.shape[0]
        
        # Lấy các mẫu thuộc lớp c
        X_train_c = X_train[y_train == c]
        
        # Tính xác suất hậu nghiệm cho từng pixel
        likelihood[c, :] = (np.sum(X_train_c, axis=0) + 1) / (np.sum(X_train_c) + num_pixels)
    
    return prior, likelihood

