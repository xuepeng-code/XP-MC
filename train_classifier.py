import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from models import MLP_gene, MLP_protein, Model_all, Classifier_multi, Classifier_MRI,FeatureProjector,ResNet_3D,Bottleneck,BasicBlock,ResNet34_3D
from dataset import CustomDatasetWithLabels
from losses import NTXentLoss
import os
import pandas as pd
from tqdm import tqdm
import csv
import numpy as np
import nibabel as nib
# import matplotlib as plt
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
from scipy.spatial.distance import cosine
from sklearn.cluster import KMeans
from matplotlib.colors import ListedColormap
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.decomposition import PCA
from semopy import Model
from sklearn.preprocessing import StandardScaler
from semopy import calc_stats
import shutil
from losses import NTXentLoss
from sklearn.model_selection import StratifiedKFold
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


import numpy as np
import os
from sklearn.preprocessing import normalize
from scipy.spatial.distance import cosine
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from matplotlib import cm
import numpy as np
import os
from sklearn.preprocessing import normalize
from scipy.spatial.distance import cosine
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.lines import Line2D
import numpy as np
import os
from sklearn.preprocessing import normalize
from scipy.spatial.distance import cosine
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.lines import Line2D



path_coefficients_csv = 'result/path_coefficients.csv'
fit_indices_csv = 'result/fit_indices.csv'

def save_metrics_to_csv(metrics, filename):
    metrics_df = pd.DataFrame([metrics])

    if not os.path.isfile(filename):
        metrics_df.to_csv(filename, index=True)
    else:
        metrics_df.to_csv(filename, mode='a', header=False, index=False)
def save_region_features_to_csv(region_features_df, all_ids, epoch, filename='result/classifier/region_features.csv'):
    region_features_df = pd.concat([pd.Series(all_ids, name='ID'), region_features_df], axis=1)
    region_features_df.to_csv(filename, mode='a', header=not epoch, index=False)
def save_loss_to_csv(filename, epoch,avg_loss_all,avg_loss_one):
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        if epoch == 0:
            writer.writerow(['Epoch', 'Loss_ALL', 'Loss_ONE']) 
        writer.writerow([epoch + 1, avg_loss_all, avg_loss_one])

def evaluate_classifier(cnn_encoder, snp_model, protein_model, model, classifier, Classifier_one, feature3,criterion,criterion1,val_loader, device,epoch, phase='val'):
    classifier.eval()
    Classifier_one.eval()
    cnn_encoder.eval()
    snp_model.eval()
    protein_model.eval()
    model.eval()
    feature3.eval()
    all_labels = []
    all_preds = []
    all_pred_probs = []
    total_loss_all = 0
    with torch.no_grad():
        # mask_tensor = load_nifti_as_tensor(aal_file_path)
        for mri_data, data1, data2,data3, mask, labels in tqdm(val_loader, desc="Validation"):
            mri_data = mri_data.to(device)
            data1 = data1.to(device) if data1 is not None else None
            data2 = data2.to(device) if data2 is not None else None
            data3 = data3.to(device) if data3 is not None else None
            labels = labels.to(device)
            # print(mask[:, 0].sum())
            if data1 is not None and mask[:, 0].sum() == 0:
                data1 = None
            if data2 is not None and mask[:, 1].sum() == 0:
                data2 = None

            mri_embedding = cnn_encoder(mri_data)
            ROI_feature = Classifier_one(data3)
            mri_embedding = mri_embedding+ROI_feature
            attended_mm, _ = model.attn11(mri_embedding, mri_embedding)
            # combined_features = mri_embedding

            if mask[:, 0].sum() > 0 and mask[:, 1].sum() > 0 and data1 is not None and data2 is not None and mri_data is not None:
                snp_output = snp_model(data1)
                protein_output = protein_model(data2)
                outputs = model(mri_embedding, snp_output, protein_output)

                attended_ms = outputs['attended_ms']
                attended_mp = outputs['attended_mp']
                attended_sp = outputs['attended_sp']
                combined_features = torch.cat([attended_ms, attended_mp, attended_sp], dim=1)
                # combined_features = feature3(combined_features_o)
                loss_ms = criterion1(attended_ms, mri_embedding)
                loss_mp = criterion1(attended_mp, mri_embedding)
                loss_sp = criterion1(attended_sp, snp_output)
                loss_att = loss_ms + loss_mp + loss_sp
                loss_1 = criterion1(mri_embedding, protein_output)
                loss_2 = criterion1(snp_output, mri_embedding)
                loss_3 = criterion1(protein_output, snp_output)  
                loss_or = loss_1 + loss_2 + loss_3
                loss = loss_att + loss_or                
                batch_size_current = mri_embedding.size(0)

            elif mask[:, 0].sum() > 0 and data1 is not None:
                snp_output = snp_model(data1)
                attended_ms, _ = model.attn12(mri_embedding, snp_output)
                combined_features = torch.cat([attended_ms, attended_ms, attended_ms], dim=1)
                loss_att = criterion1(attended_ms, mri_embedding)
                loss_or = criterion1(snp_output, mri_embedding)
                loss = loss_att + loss_or                

            elif mask[:, 1].sum() > 0 and data2 is not None:
                protein_output = protein_model(data2)
                attended_mp, _ = model.attn13(mri_embedding, protein_output)
                combined_features = torch.cat([attended_mp,attended_mp,attended_mp], dim=1)
                loss_att = criterion1(attended_mp, mri_embedding)
                loss_or = criterion1(protein_output, mri_embedding)
                loss = loss_att + loss_or
            else:
                combined_features = torch.cat([attended_mm,attended_mm,attended_mm], dim=1)
                loss = criterion1(mri_embedding, mri_embedding)+criterion1(attended_mm, attended_mm)
                # loss = criterion1(mri_embedding, mri_embedding)

            preds_ALL = classifier(combined_features)
            loss_ALL = criterion(preds_ALL, labels)+0.01*loss
            total_loss_all += loss_ALL.item()            
            preds_ALL_probs = torch.softmax(preds_ALL, dim=1)
            # preds_ONE_probs = torch.softmax(preds_ONE, dim=1)
            '''+++++++++++++++++++++++++++++'''
            # final_pred_probs = torch.max(preds_ALL_probs, preds_ALL_probs) 
            '''+++++++++++++++++++++++++++++'''
            final_pred = torch.argmax(preds_ALL_probs, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(final_pred.cpu().numpy())
            all_pred_probs.extend(preds_ALL_probs.cpu().numpy())
        avg_loss_all = total_loss_all / len(val_loader)
        avg_loss_one=0
        save_loss_to_csv('result/test_loss.csv', epoch, avg_loss_all, avg_loss_one)
    return indicate(all_labels,all_preds,all_pred_probs,epoch,phase)

def indicate(all_labels,all_preds,all_pred_probs,epoch,phase):
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_pred_probs)
    # unique_labels = np.unique(all_labels)
    # if unique_labels.min() != 0:
    #     all_labels -= unique_labels.min()
    n_classes = all_probs.shape[1]
    classes = np.arange(n_classes)
    # print(classes)
    # all_labels_binarized = label_binarize(all_labels, classes=classes)

    val_auc = roc_auc_score(all_labels, all_probs[:,1])
    val_accuracy = accuracy_score(all_labels, all_preds)
    # val_auc = roc_auc_score(all_labels, all_preds, multi_class='ovr') 
    val_precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    val_recall = recall_score(all_labels, all_preds, pos_label=1, average='binary', zero_division=0)
    val_f1 = f1_score(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        val_sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # TPR
        val_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # TNR
    else:
        val_sensitivity = 0.0
        val_specificity = 0.0

    save_dir = 'result/classifier'
    os.makedirs(save_dir, exist_ok=True)
    roc_filename = os.path.join(save_dir, f"roc_curve_{phase}_{epoch}.png")
    # print(all_probs.shape)
    # print(all_labels_binarized.shape)
    fpr, tpr, _ = roc_curve(all_labels, all_probs[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='#31859B', label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.savefig(roc_filename)
    plt.close()
    cm_filename = os.path.join(save_dir, f"confusion_matrix_{phase}_{epoch}.png")

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.savefig(cm_filename)
    plt.close()

    return {
        'accuracy': val_accuracy,
        'auc': val_auc,
        'precision': val_precision,
        'recall': val_recall,
        'f1_score': val_f1,
        'specificity': val_specificity,
        'sensitivity': val_sensitivity
    }

def train_classifier():
    num_epochs = 80
    batch_size = 10
    batch_size_eva = 1
    learning_rate = 0.001
    patience = 80  
    best_val_accuracy = 0.0  
    patience_counter = 0
    loss_csv = 'result/training_loss.csv'
    metrics_csv = 'result/classifier_metrics.csv'
    metrics_csv_train = 'result/train_classifier_metrics.csv'
    region_features_csv = 'result/classifier/region_features.csv'
    dataset = CustomDatasetWithLabels(labels_dir, mri_dir, csv_file1, csv_file2,csv_file3)
    train_size = int(0.75 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4,drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size_eva, shuffle=False, num_workers=4)
    
    if dataset.data1 is not None:
        num_features1 = dataset.data1.shape[1]
    else:
        num_features1 = 0
    if dataset.data2 is not None:
        num_features2 = dataset.data2.shape[1]
    else:
        num_features2 = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cnn_encoder = ResNet34_3D(BasicBlock, [2, 2, 2, 2], 113,137,113).to(device)
    snp_model = MLP_gene(input_size=num_features1).to(device)
    protein_model = MLP_protein(input_size=num_features2).to(device)
    # Mask_model = Mask_project().to(device)
    model = Model_all().to(device)
    # cnn_encoder.load_state_dict(torch.load('Pre_model_save/cnn_encoder_best.pth', weights_only=True))
    # snp_model.load_state_dict(torch.load('Pre_model_save/snp_model_best.pth', weights_only=True))
    # protein_model.load_state_dict(torch.load('Pre_model_save/protein_model_best.pth', weights_only=True))
    # model.load_state_dict(torch.load('Pre_model_save/attention_model_best.pth', weights_only=True))
    Classifier_all = Classifier_multi(num_classes=2).to(device)
    Classifier_one = Classifier_MRI(num_classes=128).to(device)
    feature3 = FeatureProjector(output_size=128).to(device)

    criterion = nn.CrossEntropyLoss()
    criterion1 = NTXentLoss(temperature=0.07)
    optimizer = torch.optim.Adam(
        list(feature3.parameters()) + 
        list(Classifier_all.parameters()) +
        list(cnn_encoder.parameters()) +
        list(snp_model.parameters()) +
        list(protein_model.parameters()) +
        list(model.parameters())+
        list(Classifier_one.parameters()),
        lr=learning_rate,
        weight_decay=1e-4  
    )

    scheduler_all = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        factor=0.1, 
        patience=100
    )

    open(loss_csv, 'w').close()
    open(metrics_csv, 'w').close()
    open(metrics_csv_train, 'w').close()
    for epoch in range(num_epochs):
        Classifier_all.train()
        Classifier_one.train()
        cnn_encoder.train()
        snp_model.train()
        protein_model.train()
        model.train()
        feature3.train()
        all_labels = []
        all_preds = []
        all_pred_probs = []
        total_loss_all = 0
        total_loss_one = 0
        all_mri_features = []
        all_snp_features = []
        all_protein_features = []

        for batch_idx, (mri_data, data1, data2, data3, mask, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            mri_data, labels = mri_data.to(device), labels.to(device).long()
            data1 = data1.to(device)
            data2 = data2.to(device)
            data3 = data3.to(device)
            mask = mask.to(device)  # Shape: (batch_size, 2)

            mri_embedding = cnn_encoder(mri_data)
            ROI_feature = Classifier_one(data3)
            mri_embedding = mri_embedding+ROI_feature
            attended_mm, _ = model.attn11(mri_embedding, mri_embedding)
            # combined_features = mri_embedding

            if mask[:, 0].sum() > 0 and mask[:, 1].sum() > 0 and data1 is not None and data2 is not None and mri_data is not None:
                snp_output = snp_model(data1)
                protein_output = protein_model(data2)
                outputs = model(mri_embedding, snp_output, protein_output)

                attended_ms = outputs['attended_ms']
                attended_mp = outputs['attended_mp']
                attended_sp = outputs['attended_sp']
                combined_features = torch.cat([attended_ms, attended_mp, attended_sp], dim=1)
                # combined_features = feature3(combined_features_o)

                loss_ms = criterion1(attended_ms, attended_mp)
                loss_mp = criterion1(attended_mp, attended_sp)
                loss_sp = criterion1(attended_sp, attended_ms)
                loss_att = (loss_ms + loss_mp + loss_sp)/3
                loss_1 = criterion1(mri_embedding, protein_output)
                loss_2 = criterion1(snp_output, mri_embedding)
                loss_3 = criterion1(protein_output, snp_output)  
                loss_or = (loss_1 + loss_2 + loss_3)/3
                loss = loss_att + loss_or                
                batch_size_current = mri_embedding.size(0)

                for i in range(batch_size_current):
                    mri_feature = mri_embedding[i].detach().cpu().numpy()
                    all_mri_features.append(mri_feature)
                    snp_feature = snp_output[i].detach().cpu().numpy()
                    all_snp_features.append(snp_feature)
                    protein_feature = protein_output[i].detach().cpu().numpy()
                    all_protein_features.append(protein_feature)
                # print(11111)
            elif mask[:, 0].sum() > 0 and data1 is not None:
                snp_output = snp_model(data1)
                attended_ms, _ = model.attn12(mri_embedding, snp_output)
                combined_features = torch.cat([attended_ms, attended_ms, attended_ms], dim=1)
                loss_att = criterion1(attended_ms, mri_embedding)
                loss_or = criterion1(snp_output, mri_embedding)
                loss = loss_att + loss_or                
                # print(22222)
            elif mask[:, 1].sum() > 0 and data2 is not None:
                protein_output = protein_model(data2)
                attended_mp, _ = model.attn13(mri_embedding, protein_output)
                combined_features = torch.cat([attended_mp, attended_mp, attended_mp], dim=1)
                loss_att = criterion1(attended_mp, mri_embedding)
                loss_or = criterion1(protein_output, mri_embedding)
                loss = loss_att + loss_or
            elif mask[:, 1].sum() == 0 and mask[:, 0].sum() == 0:
                combined_features = torch.cat([attended_mm, attended_mm, attended_mm], dim=1)
                loss = criterion1(mri_embedding, mri_embedding)+criterion1(attended_mm, attended_mm)
                # loss = criterion1(mri_embedding, mri_embedding)
            preds_ALL = Classifier_all(combined_features)
            # print(loss)
            loss_ALL = criterion(preds_ALL, labels)+0.1*loss
            optimizer.zero_grad()
            loss_ALL.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss_all += loss_ALL.item()
            preds_ALL_probs = torch.softmax(preds_ALL, dim=1)  

            '''+++++++++++++++++++++++++++++'''
            final_pred = torch.argmax(preds_ALL_probs, dim=1)
            all_labels.extend(labels.detach().cpu().numpy())
            all_preds.extend(final_pred.detach().cpu().numpy())
            all_pred_probs.extend(preds_ALL_probs.detach().cpu().numpy())
        avg_loss_all = total_loss_all / len(train_loader)
        avg_loss_one = total_loss_one / len(train_loader)
        save_loss_to_csv(loss_csv, epoch, avg_loss_all, avg_loss_one)

        '''++++++++++++++++++++++++'''

        metrics = evaluate_classifier(cnn_encoder, snp_model, protein_model, model, Classifier_all, Classifier_one, feature3,criterion,criterion1,val_loader, device,epoch, phase='val')
        save_metrics_to_csv(metrics, metrics_csv)
        metrics_train = indicate(all_labels,all_preds,all_pred_probs,epoch,phase='train')
        save_metrics_to_csv(metrics_train, metrics_csv_train)
        scheduler_all.step(metrics['accuracy'])
        # scheduler_one.step(metrics['accuracy'])
        if metrics['accuracy'] > best_val_accuracy:
            best_val_accuracy = metrics['accuracy']
            patience_counter = 0
            torch.save(Classifier_all.state_dict(), 'model_save/classifier_all_best.pth')
            torch.save(Classifier_one.state_dict(), 'model_save/classifier_one_best.pth')
            torch.save(cnn_encoder.state_dict(), 'model_save/cnn_encoder_best.pth')
            torch.save(snp_model.state_dict(), 'model_save/snp_model_best.pth')
            torch.save(protein_model.state_dict(), 'model_save/protein_model_best.pth')
            torch.save(model.state_dict(), 'model_save/attention_model_best.pth')
        elif metrics['accuracy'] == best_val_accuracy:
            torch.save(Classifier_all.state_dict(), 'model_save/classifier_all_best.pth')
            torch.save(Classifier_one.state_dict(), 'model_save/classifier_one_best.pth')
            torch.save(cnn_encoder.state_dict(), 'model_save/cnn_encoder_best.pth')
            torch.save(snp_model.state_dict(), 'model_save/snp_model_best.pth')
            torch.save(protein_model.state_dict(), 'model_save/protein_model_best.pth')
            torch.save(model.state_dict(), 'model_save/attention_model_best.pth')            
            patience_counter += 1
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stop: if there is no improvement in the {epoch+1} round, the training will be terminated in advance.")
            break
        print(f"Epoch {epoch+1}/{num_epochs}, Loss_all: {avg_loss_all:.6f}, Loss_one: {avg_loss_one:.6f}, Accuracy_test: {metrics['accuracy']:.6f},Accuracy_train: {metrics_train['accuracy']:.6f}")
        torch.save(Classifier_all.state_dict(), 'model_save/classifier_all_final.pth')
        torch.save(Classifier_one.state_dict(), 'model_save/classifier_one_final.pth')
        torch.save(cnn_encoder.state_dict(), 'model_save/cnn_encoder_final.pth')
        torch.save(snp_model.state_dict(), 'model_save/snp_model_final.pth')
        torch.save(protein_model.state_dict(), 'model_save/protein_model_final.pth')
        torch.save(model.state_dict(), 'model_save/attention_model_final.pth')
    print("finished!")



if __name__ == "__main__":
    mri_dir = "....../train/MRI"
    csv_file1 = "....../train/GENE_DATA.csv"
    csv_file2 = "....../train/PROTEIN_DATA.csv"
    csv_file3 = '....../train/ROI.csv'
    # csv_file1 = None
    # csv_file2 = None
    labels_dir = "....../train/Label.csv"
    distance_file = 'result/classifier/distance.csv'
    save_path = 'result/classifier'
    model_save_path = 'model_save'
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:
        shutil.rmtree(save_path)

    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    train_classifier()