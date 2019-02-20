import numpy as np
from pandas import DataFrame
import pandas as pd
import numpy as np
trian_txt_path = r'C:\Users\zugle\Desktop\Python_code\steam\zhengqi_train.txt'
test_txt_path = r'C:\Users\zugle\Desktop\Python_code\steam\zhengqi_test.txt'
dict_data = {}


def read_train_text():
    with open(trian_txt_path, 'r') as traintxt_to_read:
        file_data = pd.read_csv(trian_txt_path, delim_whitespace=True)  #空格为分隔符
        return file_data


def read_test_test():
    with open(test_txt_path, 'r') as testtxt_to_read:
        file_data = pd.read_csv(test_txt_path, delim_whitespace=True)
        return file_data


def linear_model():
    train_data = np.mat(np.array(read_train_text()))
    data_x = np.hstack((np.ones((train_data.shape[0], 1), float), train_data[:, 0:-1]))
    data_x_t = np.transpose(data_x)
    data_b = np.linalg.solve(np.dot(data_x_t, data_x), np.dot(data_x_t, train_data[:, -1]))
    test_data = np.mat(np.array(read_test_test()))
    t_data_x = np.hstack((np.ones((test_data.shape[0], 1), float), test_data))
    # print(np.dot(t_data_x, np.mat(data_b)))
    np.savetxt('linear_model.txt', np.dot(t_data_x, np.mat(data_b)), newline='\r\n', delimiter=' ')
    return np.dot(t_data_x, np.mat(data_b))


def save_answer(answer):
    np.savetxt('linear_model.txt', answer, newline='\n', delimiter=' ')


def main():
    answer = linear_model()
    # save_answer(answer)


if __name__ == '__main__':
    main()




