import os
import pandas as pd
import shutil
import random
def ensure_empty_dir(path):

    if os.path.exists(path):
        if os.path.isdir(path):

            shutil.rmtree(path)
        else:
            raise ValueError(f"Path exists but is not a folder: {path}")
    os.makedirs(path, exist_ok=True)
mri_dir = "....../MRI"  
csv_file1 = "....../GENE_DATAF.csv"  
csv_file2 = "....../PROTEIN_DATA.csv"  
csv_file3 = '....../ROI1.csv'
Label = "....../Label1.csv"  

data01_path = 'data/data01' # NC and MCI
data02_path = 'data/data02' # NC and AD
data12_path = 'data/data12' # MCI and AD

ensure_empty_dir(data01_path)
ensure_empty_dir(data02_path)
ensure_empty_dir(data12_path)
mri_files = [f for f in os.listdir(mri_dir) if f.endswith(".nii")]
random.shuffle(mri_files)
mri_ids = [os.path.splitext(os.path.splitext(f)[0])[0] for f in mri_files]
data1 = pd.read_csv(csv_file1)
data2 = pd.read_csv(csv_file2)
data3 = pd.read_csv(Label)
data4 = pd.read_csv(csv_file3)
data1_filtered = data1[data1['ID'].isin(mri_ids)]
data2_filtered = data2[data2['ID'].isin(mri_ids)]
data3_filtered = data3[data3['ID'].isin(mri_ids)]
data4_filtered = data4[data4['ID'].isin(mri_ids)]
group_01_ids = data3_filtered[(data3_filtered['Label'] == 0) | (data3_filtered['Label'] == 1)]['ID'].tolist()
group_02_ids = data3_filtered[(data3_filtered['Label'] == 0) | (data3_filtered['Label'] == 2)]['ID'].tolist()
group_12_ids = data3_filtered[(data3_filtered['Label'] == 1) | (data3_filtered['Label'] == 2)]['ID'].tolist()
group_01_mri_files = [f for f in mri_files if os.path.splitext(os.path.splitext(f)[0])[0] in group_01_ids]
group_02_mri_files = [f for f in mri_files if os.path.splitext(os.path.splitext(f)[0])[0] in group_02_ids]
group_12_mri_files = [f for f in mri_files if os.path.splitext(os.path.splitext(f)[0])[0] in group_12_ids]
def split_data(mri_files, data1_filtered, data2_filtered, data3_filtered, data4_filtered, group_name):
    num_files = len(mri_files)
    num_train = int(num_files * 0.8)  
    num_val = num_files - num_train  

    random.shuffle(mri_files)
    train_mri_files = mri_files[:num_train]
    val_mri_files = mri_files[num_train:]
    train_data1 = data1_filtered[data1_filtered['ID'].isin([os.path.splitext(os.path.splitext(f)[0])[0] for f in train_mri_files])]
    val_data1 = data1_filtered[data1_filtered['ID'].isin([os.path.splitext(os.path.splitext(f)[0])[0] for f in val_mri_files])]

    train_data2 = data2_filtered[data2_filtered['ID'].isin([os.path.splitext(os.path.splitext(f)[0])[0] for f in train_mri_files])]
    val_data2 = data2_filtered[data2_filtered['ID'].isin([os.path.splitext(os.path.splitext(f)[0])[0] for f in val_mri_files])]

    train_data3 = data3_filtered[data3_filtered['ID'].isin([os.path.splitext(os.path.splitext(f)[0])[0] for f in train_mri_files])]
    val_data3 = data3_filtered[data3_filtered['ID'].isin([os.path.splitext(os.path.splitext(f)[0])[0] for f in val_mri_files])]

    train_data4 = data4_filtered[data4_filtered['ID'].isin([os.path.splitext(os.path.splitext(f)[0])[0] for f in train_mri_files])]
    val_data4 = data4_filtered[data4_filtered['ID'].isin([os.path.splitext(os.path.splitext(f)[0])[0] for f in val_mri_files])]

    train_mri_dir = os.path.join(group_name, 'train/MRI')
    val_mri_dir = os.path.join(group_name, 'val/MRI')
    os.makedirs(train_mri_dir, exist_ok=True)
    os.makedirs(val_mri_dir, exist_ok=True)

    for mri_file in train_mri_files:
        shutil.copy(os.path.join(mri_dir, mri_file), os.path.join(train_mri_dir, mri_file))
    for mri_file in val_mri_files:
        shutil.copy(os.path.join(mri_dir, mri_file), os.path.join(val_mri_dir, mri_file))

    train_data1.to_csv(os.path.join(group_name, 'train', 'GENE_DATA.csv'), index=False)
    val_data1.to_csv(os.path.join(group_name, 'val', 'GENE_DATA.csv'), index=False)

    train_data2.to_csv(os.path.join(group_name, 'train', 'PROTEIN_DATA.csv'), index=False)
    val_data2.to_csv(os.path.join(group_name, 'val', 'PROTEIN_DATA.csv'), index=False)

    train_data3.to_csv(os.path.join(group_name, 'train', 'Label.csv'), index=False)
    val_data3.to_csv(os.path.join(group_name, 'val', 'Label.csv'), index=False)

    train_data4.to_csv(os.path.join(group_name, 'train', 'ROI.csv'), index=False)
    val_data4.to_csv(os.path.join(group_name, 'val', 'ROI.csv'), index=False)



split_data(group_01_mri_files, data1_filtered, data2_filtered, data3_filtered, data4_filtered, data01_path)
split_data(group_02_mri_files, data1_filtered, data2_filtered, data3_filtered, data4_filtered, data02_path)
split_data(group_12_mri_files, data1_filtered, data2_filtered, data3_filtered, data4_filtered, data12_path)


