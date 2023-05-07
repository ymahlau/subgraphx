from src.utils.logging import save_data, load_data
import time

path = './unit_test/utils/test_data2'
dict_test = {(0, 1): [], (1, 2): []}
test_tuple = ({0, 2, 3, 4}, 0.1, 0.2, 125.7)
dict_test[(0, 1)] = dict_test[(0, 1)] + [test_tuple]

save_data(path, dict_test)
new_dict = load_data(path)
print(new_dict)


