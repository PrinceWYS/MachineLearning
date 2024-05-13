import argparse
import pickle
from libsvm.svmutil import *
import SVM

def load_data(data_path):
    # load data
    print(f'[INFO] Loading data from {data_path}')
    data = pickle.load(open(data_path, 'rb'))
    keys = data.keys()
    print(f'[INFO] Data has rows: {len(data[keys])}, columns: {len(keys)}')
    y = data[keys[1]].tolist()
    Y = [1 if i == 'M' else -1 for i in y]
    X = []
    for index, row in data.iterrows():
        x = []
        for key in keys:
            if key != keys[1] and key != keys[0]:
                x.append(row[key])
        X.append(x)
    print(f'[INFO] Data loaded successfully!')
    return X, Y

def test_libsvm(X_train, Y_train, X_test, Y_test):
    prob = svm_problem(Y_train, X_train)
    param = svm_parameter('-t 0 -b 1 -q')
    model = svm_train(prob, param)
    p_label, p_acc, p_val = svm_predict(Y_test, X_test, model)
    print(f'[INFO] Accuracy: {p_acc[0]}%, MSE: {p_acc[1]}, SCC: {p_acc[2]}')
    print(model.nSV)
    
def test_my_svm(X_train, Y_train, X_test, Y_test):
    model = SVM.SVM(X_train, Y_train)
    model.fit()
    model.predict(X_test, Y_test)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data/data.pkl')
    args = parser.parse_args()
    data_path = args.data_path
    
    # load data
    X, Y = load_data(data_path)
    
    # using former 500 samples as training data, others as testing data
    X_train = X[:500]
    Y_train = Y[:500]
    X_test = X[500:]
    Y_test = Y[500:]
    
    # test my own SVM
    test_my_svm(X_train, Y_train, X_test, Y_test)
    # test libsvm
    # test_libsvm(X_train, Y_train, X_test, Y_test)