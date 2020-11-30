from sklearn.model_selection import train_test_split
import glob
from shutil import copyfile
import json

def create_json_dataset(dest_folder, split_part, files, log_progress=True):
    all_data = []

    for i, json_path in enumerate(files):
        with open(json_path) as json_file:
            data = json.load(json_file)
            if len(data) > 0:
                data = data[0]
            else:
                continue

        img_id = json_path.split('/')[-1].split('.')[0]
        data['id'] = img_id
        all_data.append(data)

        if i % 50 == 0 and log_progress:
            print(f'{split_part}: {i}')

    with open(f'{dest_folder}/{split_part}.json', 'w') as outfile:
        json.dump(all_data, outfile)

if __name__ == "__main__":
    jsons_files = glob.glob('../ffhq-features-dataset/json/*.json')

    X_train, X_test, _, _ = train_test_split(jsons_files, jsons_files, test_size=0.2, random_state=1)

    X_train, X_val, _, _ = train_test_split(X_train, X_train, test_size=0.25, random_state=1)

    dest_folder = 'D:/Work/GAN/FFHQ'

    create_json_dataset(dest_folder, 'train', X_train)
    create_json_dataset(dest_folder, 'val', X_val)
    create_json_dataset(dest_folder, 'test', X_test)