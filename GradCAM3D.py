import torch
import torch.nn as nn
from torchvision import models
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from captum.attr import LayerGradCam
from captum.attr import visualization as viz
import os
from scipy.ndimage import zoom
from models import MLP_gene, MLP_protein, Model_all, Classifier_multi, Classifier_MRI, FeatureProjector, ResNet_3D, Bottleneck, BasicBlock, ResNet34_3D
import pandas as pd

class LayerActivations:
    features = None
    
    def __init__(self, model, layer):
        self.hook = layer.register_forward_hook(self.hook_fn)
        
    def hook_fn(self, module, input, output):
        self.features = output.detach().cpu()
        
    def remove(self):
        self.hook.remove()

def load_mri_image(nii_path):

    img = nib.load(nii_path)
    img_data = img.get_fdata()


    img_data = (img_data - np.mean(img_data)) / np.std(img_data)
    img_data = img_data.astype(np.float32)
    img_data = np.expand_dims(img_data, axis=0)  # C x D x H x W
    img_tensor = torch.tensor(img_data)
    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor

def load_model(model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):

    model = ResNet34_3D(BasicBlock, [2, 2, 2, 2], 113, 137, 113)  
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 128)  
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def compute_gradcam(model, input_tensor, target_layer, target_class=None):
    device = input_tensor.device
    grad_cam = LayerGradCam(model, target_layer)

    if target_class is None:
        output = model(input_tensor)
        target_class = output.argmax(dim=1).item()

    attributions = grad_cam.attribute(input_tensor, target=target_class)
    attributions = attributions.squeeze().cpu().detach().numpy()
    return attributions

def load_atlas(atlas_path, label_dict_path):

    atlas_nii = nib.load(atlas_path)
    atlas_data = atlas_nii.get_fdata().astype(int)

    label_df = pd.read_csv(label_dict_path)
    label_dict = {int(row['label_id']): row['region_name'] for _, row in label_df.iterrows()}
    return atlas_data, label_dict

def identify_top_brain_regions(attributions_upsampled, atlas_data, label_dict, top_n=5):
    labels_in_atlas = np.unique(atlas_data)
    labels_in_atlas = labels_in_atlas[labels_in_atlas != 0]
    
    brain_region_scores = {}
    for label_id in labels_in_atlas:
        mask = (atlas_data == label_id)
        region_values = attributions_upsampled[mask]
        region_score = np.max(region_values) if region_values.size > 0 else 0
        brain_region_scores[label_id] = region_score

    sorted_regions = sorted(brain_region_scores.items(), key=lambda x: x[1], reverse=True)
    top_regions = [(label_dict.get(label_id, f"Unknown-{label_id}"), score)
                   for label_id, score in sorted_regions[:top_n]]
    return top_regions
plt.rcParams['font.size'] = 24
plt.rcParams['axes.titlesize'] = 24
plt.rcParams['axes.labelsize'] = 24
plt.rcParams['xtick.labelsize'] = 24
plt.rcParams['ytick.labelsize'] = 24
plt.rcParams['legend.fontsize'] = 24
plt.rcParams['figure.titlesize'] = 26

def visualize_gradcam_three_directions(attributions, original_image, slice_indices=None, save_path=None):

    import matplotlib.pyplot as plt
    from scipy.ndimage import zoom

    C, D_orig, H_orig, W_orig = original_image.shape
    D_attr, H_attr, W_attr = attributions.shape

    print(f"Original image shape: {original_image.shape}")
    print(f"Attributions shape: {attributions.shape}")

    if D_attr != D_orig or H_attr != H_orig or W_attr != W_orig:
        zoom_factors = (D_orig / D_attr, H_orig / H_attr, W_orig / W_attr)
        print(f"zoom_factors: {zoom_factors}")
        attributions_upsampled = zoom(attributions, zoom_factors, order=1)
        print(f"attributions shape: {attributions_upsampled.shape}")
    else:
        attributions_upsampled = attributions

    if slice_indices is None:
        slice_indices = {
            'axial': D_orig // 2,
            'sagittal': W_orig // 2,
            'coronal': H_orig // 2
        }

    axial_slice = original_image[0, slice_indices['axial'], :, :]
    sagittal_slice = original_image[0, :, :, slice_indices['sagittal']]
    coronal_slice = original_image[0, :, slice_indices['coronal'], :]

    cam_axial = attributions_upsampled[slice_indices['axial'], :, :]
    cam_sagittal = attributions_upsampled[:, :, slice_indices['sagittal']]
    cam_coronal = attributions_upsampled[:, slice_indices['coronal'], :]

    cam_sagittal = np.transpose(cam_sagittal)
    sagittal_slice = np.transpose(sagittal_slice)

    cam_coronal = np.transpose(cam_coronal)
    coronal_slice = np.transpose(coronal_slice)

    def normalize_cam(cam_slice):
        cam_slice = cam_slice - np.min(cam_slice)
        if np.max(cam_slice) != 0:
            cam_slice = cam_slice / np.max(cam_slice)
        return cam_slice

    cam_axial = normalize_cam(cam_axial)
    cam_sagittal = normalize_cam(cam_sagittal)
    cam_coronal = normalize_cam(cam_coronal)



    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(axial_slice, cmap='gray')
    im0 = axes[0].imshow(cam_axial, cmap='jet', alpha=0.5)
    axes[0].set_title('Axial Slice with Grad-CAM')
    axes[0].axis('off')

    axes[1].imshow(sagittal_slice, cmap='gray')
    im1 = axes[1].imshow(cam_sagittal, cmap='jet', alpha=0.5)
    axes[1].set_title('Sagittal Slice with Grad-CAM')
    axes[1].axis('off')

    axes[2].imshow(coronal_slice, cmap='gray')
    im2 = axes[2].imshow(cam_coronal, cmap='jet', alpha=0.5)
    axes[2].set_title('Coronal Slice with Grad-CAM')
    axes[2].axis('off')

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    norm = plt.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(cmap='jet', norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, label='Grad-CAM Intensity')

    plt.tight_layout(rect=[0, 0, 0.9, 1])

    if save_path is not None:
        plt.savefig(save_path)
        plt.close(fig)
    else:
        plt.show()

def main():

    model_path = '.../model_save/cnn_encoder_final.pth'
    nii_directory = 'explaindata/MCI/MRI'   
    atlas_path = 'data/1/neuromorphometrics.nii'
    label_dict_path = 'data/1/neuromorphometrics.csv'
    out_csv_path = 'explain_result/all_brain_regions_row_name.csv'
    out_fig_dir = 'explain_result/GradCAM'
    os.makedirs(out_fig_dir, exist_ok=True)

    top_n = 50
    target_layer_name = 'layer2'  

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(nii_directory):
        raise FileNotFoundError(f"NIfTI directory not found: {nii_directory}")
    if not os.path.exists(atlas_path):
        raise FileNotFoundError(f"Atlas file not found: {atlas_path}")
    if not os.path.exists(label_dict_path):
        raise FileNotFoundError(f"Label dictionary file not found: {label_dict_path}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = load_model(model_path, device=device)
    target_layer = getattr(model, target_layer_name)  # model.layer2
    atlas_data, label_dict = load_atlas(atlas_path, label_dict_path)
    all_region_names = list(label_dict.values())
    all_results_df = pd.DataFrame(index=all_region_names, dtype=float)
    for file_name in os.listdir(nii_directory):
        if not (file_name.endswith('.nii') or file_name.endswith('.nii.gz')):
            continue
        nii_path = os.path.join(nii_directory, file_name)
        ID = os.path.splitext(os.path.splitext(file_name)[0])[0]
        print(f"\nProcessing file: {file_name} => ID={ID}")
        input_tensor = load_mri_image(nii_path).to(device)
        attributions = compute_gradcam(model, input_tensor, target_layer)
        img_nib = nib.load(nii_path)
        img_data = img_nib.get_fdata()
        img_data = (img_data - np.mean(img_data)) / np.std(img_data)
        img_data = img_data.astype(np.float32)
        img_data = np.expand_dims(img_data, axis=0)  # [C, D, H, W]
        fig_save_path = os.path.join(out_fig_dir, f"{ID}_gradcam.png")
        visualize_gradcam_three_directions(attributions, img_data, save_path=fig_save_path)
        print(f"Grad-CAM figure saved to {fig_save_path}")

        if attributions.shape != atlas_data.shape:
            zoom_factors = (atlas_data.shape[0] / attributions.shape[0],
                            atlas_data.shape[1] / attributions.shape[1],
                            atlas_data.shape[2] / attributions.shape[2])
            print("Grad-CAM and Atlas shape is not same, processing...")
            attributions_upsampled = zoom(attributions, zoom_factors, order=1)
        else:
            attributions_upsampled = attributions

        top_regions = identify_top_brain_regions(
            attributions_upsampled, atlas_data, label_dict, top_n=top_n
        )
        print(f"Top {top_n} brain regions for ID={ID}:")
        for i, (rname, rscore) in enumerate(top_regions, start=1):
            print(f"{i}. {rname} => {rscore:.8f}")
        all_results_df[ID] = np.nan
        for region_name, score in top_regions:
            if region_name in all_results_df.index:
                all_results_df.at[region_name, ID] = score

    all_results_df=all_results_df.T
    all_results_df.to_csv(out_csv_path)



if __name__ == '__main__':
    main()
