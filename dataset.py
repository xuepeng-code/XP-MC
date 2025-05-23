import torch
from torch.utils.data import Dataset
import pandas as pd
import nibabel as nib  # 用于读取MRI图像（NIfTI格式）
import os

class CustomDataset(Dataset):
    def __init__(self, mri_dir, csv_file1=None, csv_file2=None, transform=None):
        """
        mri_dir: MRI
        csv_file1: data1,CSV file
        csv_file2: data2),CSV file
        transform: 
        """
        self.mri_dir = mri_dir
        # self.transform = transform
        
        # MRI
        self.mri_files = [f for f in os.listdir(mri_dir) if f.endswith('.nii')]
        self.ids = [os.path.splitext(os.path.splitext(f)[0])[0] for f in self.mri_files]  # 提取ID


        if csv_file1 is not None:
            self.data1 = pd.read_csv(csv_file1)
            self.data1 = self.data1.set_index('ID')
            self.data1_mean = torch.tensor(self.data1.mean().values, dtype=torch.float32)
            self.data1_std = torch.tensor(self.data1.std().values + 1e-8, dtype=torch.float32)
            self.data1_shape = self.data1.iloc[0].shape
        else:
            self.data1 = None
            self.data1_shape = None
            self.data1_mean = None
            self.data1_std = None


        if csv_file2 is not None:
            self.data2 = pd.read_csv(csv_file2)
            self.data2 = self.data2.set_index('ID')
            self.data2_mean = torch.tensor(self.data2.mean().values, dtype=torch.float32)
            self.data2_std = torch.tensor(self.data2.std().values + 1e-8, dtype=torch.float32)
            self.data2_shape = self.data2.iloc[0].shape
        else:
            self.data2 = None
            self.data2_shape = None
            self.data2_mean = None
            self.data2_std = None

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        
        id = self.ids[idx]


        mri_path = os.path.join(self.mri_dir, f"{id}.nii")  # NIfTI格式
        mri_image = nib.load(mri_path).get_fdata()
        mri_image = torch.tensor(mri_image, dtype=torch.float32)

        # mri_image = (mri_image - mri_image.mean()) / (mri_image.std() + 1e-8)
        mri_image = mri_image.unsqueeze(0)  # 添加通道维度

        # if self.transform:
        #     mri_image = self.transform(mri_image)


        if self.data1 is not None and id in self.data1.index:
            data1 = torch.tensor(self.data1.loc[id].values, dtype=torch.float32)

            data1 = (data1 - self.data1_mean) / self.data1_std
        else:
            if self.data1_shape is not None:
                data1 = torch.full(self.data1_shape, -9.0, dtype=torch.float32)
            else:
                data1 = torch.tensor([-9.0], dtype=torch.float32)


        if self.data2 is not None and id in self.data2.index:
            data2 = torch.tensor(self.data2.loc[id].values, dtype=torch.float32)

            data2 = (data2 - self.data2_mean) / self.data2_std
        else:
            if self.data2_shape is not None:
                data2 = torch.full(self.data2_shape, -9.0, dtype=torch.float32)
            else:
                data2 = torch.tensor([-9.0], dtype=torch.float32)

        return mri_image, data1, data2
class CustomDatasetWithLabels(Dataset):
    def __init__(self, labels_dir, mri_dir, csv_file1=None, csv_file2=None, csv_file3=None, transform=None):
        """
        mri_dir: MRI
        csv_file1: data1
        csv_file2: data2
        csv_file3: data3
        transform: 
        """
        self.mri_dir = mri_dir
        self.transform = transform

        self.mri_files = [f for f in os.listdir(mri_dir) if f.endswith('.nii')]
        self.ids = [os.path.splitext(os.path.splitext(f)[0])[0] for f in self.mri_files]  # 提取ID


        if csv_file1 is not None:
            self.data1 = pd.read_csv(csv_file1)
            self.data1 = self.data1.set_index('ID')
            self.data1_shape = self.data1.iloc[0].shape
        else:
            self.data1 = None
            self.data1_shape = None


        if csv_file2 is not None:
            self.data2 = pd.read_csv(csv_file2)
            self.data2 = self.data2.set_index('ID')

            self.data2_min = self.data2.min().values
            self.data2_max = self.data2.max().values
            self.data2_shape = self.data2.iloc[0].shape
        else:
            self.data2 = None
            self.data2_shape = None
            self.data2_min = None
            self.data2_max = None
        

        if csv_file3 is not None:
            self.data3 = pd.read_csv(csv_file3)
            self.data3 = self.data3.set_index('ID')

            self.data3_min = self.data3.min().values
            self.data3_max = self.data3.max().values
            self.data3_shape = self.data3.iloc[0].shape
        else:
            self.data3 = None
            self.data3_shape = None
            self.data3_min = None
            self.data3_max = None


        if labels_dir is not None:
            self.labels = pd.read_csv(labels_dir)
            self.labels = self.labels.set_index('ID')
        else:
            raise ValueError("Label!")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id = self.ids[idx]
        mri_path = os.path.join(self.mri_dir, f"{id}.nii")
        mri_image = nib.load(mri_path).get_fdata()
        mri_image = torch.tensor(mri_image, dtype=torch.float32)
        mri_image = mri_image.unsqueeze(0)  


        if self.data1 is not None and id in self.data1.index:
            data1 = torch.tensor(self.data1.loc[id].values, dtype=torch.float32)
            mask_data1 = torch.tensor(1.0, dtype=torch.float32)
        else:
            if self.data1_shape is not None:
                data1 = torch.full(self.data1_shape, 0.0, dtype=torch.float32) 
            else:
                data1 = torch.tensor([0.0], dtype=torch.float32)
            mask_data1 = torch.tensor(0.0, dtype=torch.float32)


        if self.data2 is not None and id in self.data2.index:
            data2 = torch.tensor(self.data2.loc[id].values, dtype=torch.float32)
            data2 = (data2 - torch.tensor(self.data2_min, dtype=torch.float32)) / (torch.tensor(self.data2_max, dtype=torch.float32) - torch.tensor(self.data2_min, dtype=torch.float32))
            mask_data2 = torch.tensor(1.0, dtype=torch.float32)
        else:
            if self.data2_shape is not None:
                data2 = torch.full(self.data2_shape, 0.0, dtype=torch.float32)  
            else:
                data2 = torch.tensor([0.0], dtype=torch.float32)
            mask_data2 = torch.tensor(0.0, dtype=torch.float32)


        if self.data3 is not None and id in self.data3.index:
            data3 = torch.tensor(self.data3.loc[id].values, dtype=torch.float32)
            data3 = (data3 - torch.tensor(self.data3_min, dtype=torch.float32)) / (torch.tensor(self.data3_max, dtype=torch.float32) - torch.tensor(self.data3_min, dtype=torch.float32))
            mask_data3 = torch.tensor(1.0, dtype=torch.float32)
        else:
            if self.data3_shape is not None:
                data3 = torch.full(self.data3_shape, 0.0, dtype=torch.float32)  
            else:
                data3 = torch.tensor([0.0], dtype=torch.float32)
            mask_data3 = torch.tensor(0.0, dtype=torch.float32)


        mask = torch.stack([mask_data1, mask_data2, mask_data3])  # Shape: (3,)


        if id in self.labels.index:
            label = torch.tensor(self.labels.loc[id].values[0], dtype=torch.long)
        else:
            raise ValueError(f"ID {id} is Not in the Label.csv")

        return mri_image, data1, data2, data3, mask, label





import nibabel as nib
import torch
import numpy as np

def extract_roi_from_aal(aal_file_path, batch_size=64):


    aal_img = nib.load(aal_file_path)
    aal_data = aal_img.get_fdata()  

    unique_labels = np.unique(aal_data)
    unique_labels = unique_labels[unique_labels > 0]  

    rois = []
    

    for label in unique_labels:
        coordinates = np.argwhere(aal_data == label) 
        min_coords = coordinates.min(axis=0)  
        max_coords = coordinates.max(axis=0) 
        roi = [label, *min_coords, *max_coords]
        rois.append(roi)


    rois_tensor = torch.tensor(rois, dtype=torch.float32)
    
    
    rois_tensor = rois_tensor.unsqueeze(0).expand(batch_size, -1, -1)
    
    return rois_tensor


