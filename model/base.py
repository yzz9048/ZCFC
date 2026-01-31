import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from shared_util.logger import Logger
from model.RCADataLoader import RCADataLoader
from shared_util.data_handler import copy_batch_data, rearrange_y
from datetime import datetime
from shared_util.evaluation_metrics import fault_type_classification
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from model.encoder import Encoder
from model.decoder import ImprovedDecoder
from model.t_SNE import visualize_tsne

class FaultClassifier(nn.Module):
    def __init__(self, param_dict, meta_data, num_classes=3):
        super().__init__()
        self.encoder = Encoder(param_dict, meta_data)
        self.decoder = ImprovedDecoder(in_dim=param_dict['eff_out_dim'], num_classes=num_classes)

    def forward(self, batch_data, train_key):
        out = self.encoder(batch_data, train_key)
        fault_logits, class_logits = self.decoder(out)
        return fault_logits, class_logits, out
    
class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss
    Adapted for multi-class fault diagnosis:
    - same class (including cross-system) → positive pairs
    - different classes (including fault vs normal) → negative pairs
    """
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        Args:
            features: [batch_size, feature_dim]
            labels: [batch_size]  integer labels, same id means same semantic class
        """
        device = features.device
        features = F.normalize(features, dim=1)

        # Compute similarity matrix
        sim_matrix = torch.matmul(features, features.T) / self.temperature
        sim_exp = torch.exp(sim_matrix)

        # Mask: positive if same class but not self
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        mask_self = torch.eye(mask.shape[0], device=device)
        mask = mask * (1 - mask_self)

        # Denominator: all except self
        denom = sim_exp * (1 - mask_self)
        denom = denom.sum(dim=1, keepdim=True)

        # Numerator: only positive pairs
        pos_exp = sim_exp * mask
        pos_sum = pos_exp.sum(dim=1, keepdim=True)

        # Avoid division by zero
        loss = -torch.log((pos_sum + 1e-8) / (denom + 1e-8))
        loss = loss.mean()

        return loss
        
class BaseTrainer():
    def __init__(self, param_dict):
        self.param_dict = param_dict
        self.device = torch.device(f"cuda:{param_dict['gpu']}" if torch.cuda.is_available() else "cpu")
        self.logger = Logger(logging_level='DEBUG').logger
        self.rca_data_loader = RCADataLoader(param_dict)
        # load_data expects two dataset paths (A and B)
        self.rca_data_loader.load_data(f'{self.param_dict["dataset_path_SN"]}', f'{self.param_dict["dataset_path_TT"]}')
        self.model = FaultClassifier(param_dict, self.rca_data_loader.meta_data).to(self.device)
    @staticmethod
    def l2_normalize(x, dim, eps=1e-8):
        return F.normalize(x, dim=dim, p=2, eps=eps)
    
    @torch.no_grad()
    def compute_prototypes(self):

        prototypes = []
        
        A_label = []
        emb_A, A_normal, B_normal = [], [], []          
        with torch.no_grad():
            for batch_data in self.rca_data_loader.data_loader['train_A']:
                _, _, emb = self.model(batch_data, 'train_A')
                emb_A.extend(emb)
                y = batch_data['y'].to(self.device)
                all_zero_mask = (y.sum(dim=1) == 0)
                y = torch.argmax(y, dim=1)
                A_label.extend(torch.where(all_zero_mask, torch.tensor(0).to(self.device), y+1))
            for batch_data in self.rca_data_loader.data_loader['z-score_A']:
                _, _, emb_a = self.model(batch_data, 'z-score_A')
                A_normal.append(emb_a.mean(dim=0))
            for batch_data in self.rca_data_loader.data_loader['z-score_B']:
                _, _, emb_b = self.model(batch_data, 'z-score_B')
                B_normal.append(emb_b.mean(dim=0))
        A_label = torch.stack(A_label, dim=0)
        emb_A = torch.stack(emb_A, dim=0)

        A_normal = torch.stack(A_normal, dim=0).mean(dim=0)
        normal_centroid_A = self.l2_normalize(A_normal, 0)
        B_normal = torch.stack(B_normal, dim=0).mean(dim=0)
        normal_centroid_B = self.l2_normalize(B_normal, 0)
        # distances =  torch.cdist(F.normalize(emb_A,dim=1), normal_centroid.unsqueeze(dim=0))
        # distances = distances.squeeze(1)

        for c in range(self.param_dict["ec_fault_types"]):
            pos_mask = (A_label == c)
            prototypes.append((F.normalize(emb_A,dim=1)[pos_mask]-normal_centroid_A).mean(dim=0))

        prototypes = torch.stack(prototypes, dim=0)  # (C, D)
        
        if prototypes is None:
            return None, None
        
        return prototypes, normal_centroid_A, normal_centroid_B

    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.param_dict['lr'],
                                     weight_decay=self.param_dict['weight_decay'])

        criterion_dict = dict()
        for ent_type in self.rca_data_loader.meta_data['A']['ent_types']:
                        criterion_dict[ent_type] = torch.nn.CrossEntropyLoss()
        criterion_fault = torch.nn.BCEWithLogitsLoss()

        criterion = SupConLoss(temperature=0.1)
        beat_train_loss = float('inf')

        for epoch in range(self.param_dict['epochs']):
            self.model.train()
            train_loss = 0.0
            a_iter = iter(self.rca_data_loader.data_loader['z-score_A'])
            b_iter = iter(self.rca_data_loader.data_loader['z-score_B'])
            for batch_id, batch_data_A in enumerate(self.rca_data_loader.data_loader['train_A']):
                try:
                    batch_data_B = next(b_iter)
                except StopIteration:
                    b_iter = iter(self.rca_data_loader.data_loader['z-score_B'])
                    batch_data_B = next(b_iter)
                try:
                    batch_data_A_z = next(a_iter)
                except StopIteration:
                    a_iter = iter(self.rca_data_loader.data_loader['z-score_A'])
                    batch_data_A_z = next(a_iter)

                optimizer.zero_grad()


                # --- forward A (labeled) ---
                fault_A, class_logits_A, emb_A = self.model(batch_data_A, 'train_A')
                # emb_A: (B, F, D)

                # rearrange y into dict of ent_type -> (B*F, C)
                y_A_dict = dict()
                y = batch_data_A['y'].to(self.device)
                all_zero_mask = (y.sum(dim=1) == 0)
                y = torch.argmax(y, dim=1)
                y_A_dict[ent_type] = torch.where(all_zero_mask, torch.tensor(0).to(self.device), y+1) 

                fault_logits_A = torch.where(all_zero_mask, torch.tensor(0).to(self.device), 1)
                loss_fault = criterion_fault(fault_A.squeeze(), fault_logits_A.float())
                loss_class = criterion_dict[ent_type](class_logits_A[~all_zero_mask], y_A_dict[ent_type][~all_zero_mask]-1)
                
                A_fault = emb_A[~all_zero_mask]
                A_normal = emb_A[all_zero_mask]

                _, _, emb_A_flat = self.model(batch_data_A_z, 'z-score_A')
                A_normal = torch.concatenate([A_normal,emb_A_flat])

                _, _, emb_B_flat  = self.model(batch_data_B, 'z-score_B')
                features = torch.cat([A_normal, emb_B_flat, A_fault], dim=0)
                labels = torch.cat([
                    torch.zeros(len(A_normal)).to(self.device),        # label 0 = normal
                    torch.zeros(len(emb_B_flat)).to(self.device),        # target system normal also label 0
                    y_A_dict[ent_type][~all_zero_mask]                 # different faults: label 1, 2, 3...
                ], dim=0)


                loss_contrast = criterion(features, labels)

                loss_cls = loss_fault + 0.8*loss_class

                total_loss = loss_cls + 0.7*loss_contrast

                total_loss.backward()
                optimizer.step()

                train_loss += total_loss.item() * batch_data_A['y'].shape[0]

            train_loss /= len(self.rca_data_loader.data_loader['train_A'].dataset)
            self.logger.info(f'[{epoch}/{self.param_dict["epochs"]}] | train_loss: {train_loss:.5f}')

            # checkpoint
            if (epoch+1) % 50 == 0:
                torch.save(self.model.state_dict(), self.param_dict["model_path"]+f"{epoch+1}.pt")
        prototypes, normal_centroid_A, normal_centroid_B = self.compute_prototypes()

        if prototypes is not None:
            torch.save({'prototypes': prototypes, 
                        'normal_centroid_A': normal_centroid_A,
                        "normal_centroid_B": normal_centroid_B}, self.param_dict["prototype_path"])
            self.logger.info("Saved prototypes to " + self.param_dict["prototype_path"])


    def evaluate_rca_d3(self):
        self.model.eval()
        self.model.load_state_dict(torch.load(self.param_dict["model_path"], map_location=self.device))
        pt = torch.load(self.param_dict["prototype_path"], map_location=self.device)
        proto, normal_centroid_A, normal_centroid_B = pt["prototypes"], pt["normal_centroid_A"], pt["normal_centroid_B"]
        #proto, normal_centroid_A, normal_centroid_B = self.compute_prototypes()

        def evaluate(dataset_key, proto, normal_centroid):
            y_pred_list1, y_pred_list2 = [], []
            y_true_list = []
            zero_mask_list = []
            features = np.empty((0, 128))
            inference_time = []
            with torch.no_grad():
                for batch_id, batch_data in enumerate(self.rca_data_loader.data_loader[dataset_key]):
                    start = datetime.now().timestamp()

                    y = np.array(batch_data['y'])
                    all_zero_mask = (y.sum(axis=1) == 0)
                    y = np.argmax(y, axis=1)
                    y_true = np.where(all_zero_mask, 0, y + 1)
                    zero_mask_list.append(all_zero_mask)
                    y_true_list.append(y_true)

                    fault_logits, out_logits, emb = self.model(batch_data, dataset_key)
                    features = np.concatenate((features,emb.cpu().numpy()),0)
                    gate = torch.sigmoid(fault_logits)   # (N,)
                    mask = (gate > 0.6).cpu()
                    batch_pred = np.where(mask.squeeze(), 1, 0)
                    y_pred_list1.append(batch_pred)

                    distance = F.normalize(emb,dim=1)- normal_centroid
                    similarity = F.cosine_similarity(distance.unsqueeze(1), proto.unsqueeze(0), dim=2)
                    cls_probs = F.softmax(out_logits, dim=1)
                    sim_probs = F.softmax(similarity[:, 1:], dim=1)

                    #probs = (1-gate)*cls_probs + gate*sim_probs
                    fault = torch.argmax(sim_probs, dim=1).cpu().numpy() 
                    #batch_pred = np.where(~all_zero_mask, fault, 0)
                    y_pred_list2.append(fault[~all_zero_mask])

                    end = datetime.now().timestamp()
                    inference_time.append(end-start)
                    if (batch_id+1) % 50 ==0:
                        print("average inference time per window: ", datetime.fromtimestamp(sum(inference_time)/50))
                        inference_time = []
            y_pred1 = np.concatenate([np.asarray(x) for x in y_pred_list1], axis=0)
            y_pred2 = np.concatenate([np.asarray(x) for x in y_pred_list2], axis=0)
            y_true = np.concatenate([x for x in y_true_list], axis=0)
            zero_mask = np.concatenate([x for x in zero_mask_list], axis=0)
            visualize_tsne(features.reshape(features.shape[0],-1), y_true, dataset_key)
            return y_pred1, y_pred2, y_true, zero_mask

        # y_pred1, y_pred2, y_true, zero_mask = evaluate('test_A', proto, normal_centroid_A)
        # self.output_evaluation_rca_d3_result(y_pred1, y_pred2, y_true, zero_mask, 'test_A')

        y_pred1, y_pred2, y_true, zero_mask = evaluate('test_B', proto, normal_centroid_B)
        self.output_evaluation_rca_d3_result(y_pred1, y_pred2, y_true, zero_mask, 'test_B')


    def output_evaluation_rca_d3_result(self, y_pred1, y_pred2, y_true, zero_mask, key_name):
        """
        y_pred, y_true: np.array, shape (N,)
        """
        fault_true = np.where(~zero_mask, 1, 0)
        fault_pred = y_pred1
        precision = precision_score(fault_true, fault_pred)  # Precision = TP / (TP + FP)
        recall = recall_score(fault_true, fault_pred)      # Recall = TP / (TP + FN)
        f1 = f1_score(fault_true, fault_pred) 
        print(f"pre:{precision:.6f},rec:{recall:.6f},f1:{f1:.6f}")

        num_classes = self.param_dict.get("num_classes", 3)
        self.logger.info(f"---------- {key_name} ----------")
        print("\n Detailed reports:")
        print(
            classification_report(
                y_true[~zero_mask]-1, y_pred2,
                labels=list(range(num_classes)),
                digits=5,
                zero_division=0
            )
        )
        self.logger.info('--------------------------------')
