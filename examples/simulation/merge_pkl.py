import pickle
import os
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=99)
    args = parser.parse_args()

    data_root = '/131_data/minji/l5kit_data/action2'
    action_dict_filenames = [f for f in os.listdir(data_root) if f.endswith('pickle')]
    action_dict_filenames.sort()
    total_dict = {}
    for filename in action_dict_filenames:
        with open(os.path.join(data_root, filename), 'rb') as f:
            action_dict = pickle.load(f)
            total_dict.update(action_dict)

    with open('action2.pickle', 'wb') as f:
        pickle.dump(total_dict, f, protocol=pickle.HIGHEST_PROTOCOL)