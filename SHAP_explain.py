import nibabel as nib
import shap
import matplotlib.pyplot as plt
import torch.nn as nn
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from models import MLP_gene, MLP_protein, Model_all, Classifier_multi, Classifier_MRI,FeatureProjector,ResNet_3D,Bottleneck,BasicBlock,ResNet34_3D
from dataset import CustomDatasetWithLabels
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

class CombinedModel(nn.Module):
    def __init__(self, encoder, mlp_gene, mlp_protein, attention, projector, classifier):
        super(CombinedModel, self).__init__()
        self.encoder = encoder
        self.mlp_gene = mlp_gene
        self.mlp_protein = mlp_protein
        self.attention = attention
        self.projector = projector
        self.classifier = classifier

    def forward(self, mri, gene, protein):

        mri_embedding = self.encoder(mri)  # mri_embedding: (batch_size, 128)
        
        gene_output = self.mlp_gene(gene)       # (batch_size, 128)
        protein_output = self.mlp_protein(protein)  # (batch_size, 128)
        
        attention_outputs = self.attention(mri_embedding, gene_output, protein_output)
        attended_ms = attention_outputs['attended_ms']  # (batch_size, 128)
        attended_mp = attention_outputs['attended_mp']  # (batch_size, 128)
        attended_sp = attention_outputs['attended_sp']  # (batch_size, 128)
        
        fused_features = torch.cat([attended_ms, attended_mp, attended_sp], dim=1)  # (batch_size, 384)
        projected = self.projector(fused_features)  # (batch_size, 128)
        
        out = self.classifier(projected)  # (batch_size, num_classes)
        return out

class ModelWrapper(nn.Module):
    def __init__(self, model, device):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.device = device

    def forward(self, x):
        import numpy as np

        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float().to(self.device)
        elif isinstance(x, list):
            x = np.stack(x, axis=0)
            x = torch.from_numpy(x).float().to(self.device)
        elif isinstance(x, torch.Tensor):
            x = x.to(self.device)
        else:
            raise TypeError(f"Unknown input type: {type(x)}")

        return self.model(x)

def load_pretrained_models(device, dataset):
    encoder = ResNet34_3D(BasicBlock, [2, 2, 2, 2], 113,137,113).to(device)
    mlp_gene = MLP_gene(input_size=256).to(device)
    mlp_protein = MLP_protein(input_size=146).to(device)
    
    attention = Model_all().to(device)  # AttentionModule
    projector = FeatureProjector(output_size=128).to(device)
    classifier = Classifier_multi(num_classes=2).to(device)
    encoder.load_state_dict(torch.load('.../model_save/cnn_encoder_best.pth', map_location=device,weights_only=False))
    mlp_gene.load_state_dict(torch.load('.../model_save/snp_model_best.pth', map_location=device,weights_only=False))
    mlp_protein.load_state_dict(torch.load('.../model_save/protein_model_best.pth', map_location=device,weights_only=False))
    attention.load_state_dict(torch.load('.../model_save/attention_model_best.pth', map_location=device,weights_only=False))
    classifier.load_state_dict(torch.load('.../model_save/classifier_all_best.pth', map_location=device,weights_only=False))

    encoder.eval()
    mlp_gene.eval()
    mlp_protein.eval()
    attention.eval()
    projector.eval()
    classifier.eval()

    combined_model = CombinedModel(encoder, mlp_gene, mlp_protein, attention, projector, classifier).to(device)
    combined_model.eval()

    return combined_model, mlp_gene, mlp_protein, encoder

def get_background_data(dataloader, background_size, device):
    background_samples = []
    for i, (mri, data1, data2, data3, mask, label) in enumerate(dataloader):
        if i >= background_size:
            break
        background_samples.append((mri, data1, data2, data3, mask))

    background_mri = torch.cat([x[0] for x in background_samples], dim=0).to(device)
    background_data1 = torch.cat([x[1] for x in background_samples], dim=0).to(device)
    background_data2 = torch.cat([x[2] for x in background_samples], dim=0).to(device)
    background_data3 = torch.cat([x[3] for x in background_samples], dim=0).to(device)
    # mask = torch.cat([x[4] for x in background_samples], dim=0).to(device)  

    return background_mri, background_data1, background_data2, background_data3

def create_shap_explainers(mlp_gene, mlp_protein, background_data1, background_data2, device):

    mlp_gene_wrapper = ModelWrapper(mlp_gene, device)
    mlp_protein_wrapper = ModelWrapper(mlp_protein, device)


    explainer_mlp_gene = shap.GradientExplainer(mlp_gene_wrapper, background_data1)
    explainer_mlp_protein = shap.GradientExplainer(mlp_protein_wrapper, background_data2)

    return explainer_mlp_gene, explainer_mlp_protein

def get_test_samples(dataloader, test_size, device):
    test_samples = []
    for i, (mri, data1, data2, data3, mask, label) in enumerate(dataloader):
        if i >= test_size:
            break
        test_samples.append((mri, data1, data2, data3, mask, label))
    return test_samples

def compute_shap_values(explainer_mlp_gene, explainer_mlp_protein, test_samples, device):
    shap_values_mlp_gene = []
    shap_values_mlp_protein = []

    for sample in test_samples:
        mri, data1, data2, data3, mask, label = sample

        shap_val_mlp_gene = explainer_mlp_gene.shap_values(data1.to(device))
        shap_values_mlp_gene.append(shap_val_mlp_gene)

        shap_val_mlp_protein = explainer_mlp_protein.shap_values(data2.to(device))
        shap_values_mlp_protein.append(shap_val_mlp_protein)

    return shap_values_mlp_gene, shap_values_mlp_protein

def visualize_shap_mlp(shap_values, test_samples, dataset, sample_index=0, output_index=0, model_type='mlp_gene'):

    shap_val = shap_values[sample_index][output_index]  # Shape: (1, num_features) 或 (num_features,)
    print(f"SHAP value shape: {shap_val.shape}")
    if model_type == 'mlp_gene':
        data = test_samples[sample_index][1]  # data1
        feature_names = dataset.data1.columns.tolist()
    elif model_type == 'mlp_protein':
        data = test_samples[sample_index][2]  # data2
        feature_names = dataset.data2.columns.tolist()
    else:
        raise ValueError("model_type must be 'mlp_gene' or 'mlp_protein'")
    data = data.cpu().numpy()  # Shape: (1, num_features)
    print(f"Data shape: {data.shape}")
    if shap_val.ndim == 1:
        shap_val = shap_val.reshape(1, -1)
        print(f"Reshaped SHAP value shape: {shap_val.shape}")
    assert shap_val.shape[1] == data.shape[1], "The number of features of the shap value does not match the number of features of the data"


    shap.summary_plot(shap_val, data, feature_names=feature_names, show=False)
    plt.title(f'SHAP Summary Plot for {model_type} Output {output_index}')
    plt.show()


def aggregate_shap_values(shap_values):
    shap_abs = np.abs(shap_values)
    aggregated_shap = np.mean(shap_abs, axis=1)
    return aggregated_shap

def compute_feature_importance(shap_val, dataset, model_type='mlp_gene'):

    if model_type == 'mlp_gene':
        feature_names = dataset.data1.columns.tolist()
    elif model_type == 'mlp_protein':
        feature_names = dataset.data2.columns.tolist()
    else:
        raise ValueError("model_type must be 'mlp_gene' or 'mlp_protein'")
    
    if shap_val.ndim != 3:
        raise ValueError(f"Expected shap_val to be 3D, but got {shap_val.ndim}D")
    
    batch_size, num_features, num_outputs = shap_val.shape
    print(f"Processing SHAP values with shape: (batch_size={batch_size}, num_features={num_features}, num_outputs={num_outputs})")

    aggregated_shap = np.mean(np.abs(shap_val), axis=(0, 2))  # shape: (num_features,)
    
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': aggregated_shap
    })
    
    feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
    
    return feature_importance

def plot_feature_importance(feature_importance, top_n=20, model_type='mlp_gene'):

    plt.figure(figsize=(10, 8))

    features = feature_importance['Feature'][:top_n][::-1]     
    importances = feature_importance['Importance'][:top_n][::-1]
    bars = plt.barh(features, importances, color='skyblue')
    for bar in bars:
        width = bar.get_width()             
        y_pos = bar.get_y() + bar.get_height() / 2
        x_offset = 0.02 * importances.max()
        plt.text(width + x_offset, 
                 y_pos, 
                 f"{width:.4f}",          
                 va='center')

    plt.xlabel('Aggregated SHAP Importance')
    plt.title(f'Top {top_n} Important Features for {model_type}')
    plt.tight_layout()

    pdf_filename = f'op_{top_n}_Important_Features_for_{model_type}.pdf'
    # savepath = 
    plt.savefig(pdf_filename, bbox_inches='tight')
    print(f"Figure saved to {pdf_filename}")

    top_features_df = feature_importance.iloc[:top_n].copy()  

    csv_filename = f'op_{top_n}_Important_Features_for_{model_type}.csv'
    top_features_df.to_csv(csv_filename, index=False)
    print(f"Top {top_n} features with SHAP importance saved to {csv_filename}")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    mri_dir = 'explaindata/AD/MRI'
    labels_dir = 'explaindata/AD/Label.csv'
    snp_csv_path = 'explaindata/AD/GENE_DATA.csv'
    plasma_csv_path = 'explaindata/AD/PROTEIN_DATA.csv'
    data3_csv_path = 'explaindata/AD/ROI.csv'  

    dataset = CustomDatasetWithLabels(labels_dir=labels_dir,
                                      mri_dir=mri_dir,
                                      csv_file1=snp_csv_path,
                                      csv_file2=plasma_csv_path,
                                      csv_file3=data3_csv_path)  

    batch_size = 10
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    combined_model, mlp_gene, mlp_protein, encoder = load_pretrained_models(device, dataset)

    background_size = 50  
    background_mri, background_data1, background_data2, background_data3 = get_background_data(dataloader, background_size, device)
    explainer_mlp_gene, explainer_mlp_protein = create_shap_explainers(mlp_gene, mlp_protein, background_data1, background_data2, device)
    test_size = 1
    test_samples = get_test_samples(dataloader, test_size, device)
    shap_values_mlp_gene, shap_values_mlp_protein = compute_shap_values(
        explainer_mlp_gene, explainer_mlp_protein, test_samples, device
    )

    shap.initjs()
    sample_index = 0  

    feature_importance_gene = compute_feature_importance(
        shap_values_mlp_gene[sample_index],  # shap_val  (10, 766, 128)
        dataset,
        model_type='mlp_gene'
    )
    feature_importance_protein = compute_feature_importance(
        shap_values_mlp_protein[sample_index],  # shap_val  (10, 146, 128)
        dataset,
        model_type='mlp_protein'
    )
    plot_feature_importance(feature_importance_gene, top_n=20, model_type='mlp_gene')
    plot_feature_importance(feature_importance_protein, top_n=20, model_type='mlp_protein')


if __name__ == "__main__":
    main()

