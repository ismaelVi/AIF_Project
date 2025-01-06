import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

class OODMetrics:
    def __init__(self, scores_negatives, scores_positives):
        """
        Classe pour calculer les métriques OOD et tracer les courbes ROC.
        :param scores_negatives: Scores des échantillons négatifs (anomalies).
        :param scores_positives: Scores des échantillons positifs (normaux).
        """
        self.scores_negatives = np.array(scores_negatives)
        self.scores_positives = np.array(scores_positives)

    @staticmethod
    def confusion_matrix(scores_negatives, scores_positives, threshold):
        """
        Calcule la matrice de confusion pour un seuil donné.
        :param scores_negatives: Scores des négatifs.
        :param scores_positives: Scores des positifs.
        :param threshold: Seuil utilisé pour séparer les classes.
        :return: (fp, tp, tn, fn) : False positives, True positives, True negatives, False negatives.
        """
        false_positives = np.sum(scores_negatives > threshold)
        true_positives = np.sum(scores_positives > threshold)
        true_negatives = np.sum(scores_negatives <= threshold)
        false_negatives = np.sum(scores_positives <= threshold)
        return false_positives, true_positives, true_negatives, false_negatives

    def tpr_fpr(self, threshold):
        """
        Calcule le TPR (True Positive Rate) et le FPR (False Positive Rate) pour un seuil donné.
        :param threshold: Seuil utilisé pour séparer les classes.
        :return: (tpr, fpr)
        """
        fp, tp, tn, fn = self.confusion_matrix(self.scores_negatives, self.scores_positives, threshold)
        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)
        return tpr, fpr

    def accuracy(self, threshold):
        """
        Calcule la précision pour un seuil donné.
        :param threshold: Seuil utilisé pour séparer les classes.
        :return: Précision.
        """
        fp, tp, tn, fn = self.confusion_matrix(self.scores_negatives, self.scores_positives, threshold)
        return (tp + tn) / (tp + tn + fp + fn)

    def precision_recall(self, threshold):
        """
        Calcule la précision et le rappel pour un seuil donné.
        :param threshold: Seuil utilisé pour séparer les classes.
        :return: (précision, rappel)
        """
        fp, tp, tn, fn = self.confusion_matrix(self.scores_negatives, self.scores_positives, threshold)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        return precision, recall

    def f_beta(self, threshold, beta=1.0):
        """
        Calcule la F-beta score pour un seuil donné.
        :param threshold: Seuil utilisé pour séparer les classes.
        :param beta: Coefficient de pondération entre précision et rappel.
        :return: F-beta score.
        """
        precision, recall = self.precision_recall(threshold)
        return (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)

    def roc_auc(self):
        """
        Calcule et trace la courbe ROC.
        :return: AUROC (Area Under Receiver Operating Characteristic Curve).
        """
        # Combine scores et labels
        scores = np.concatenate((self.scores_negatives, self.scores_positives))
        labels = np.concatenate((np.zeros(len(self.scores_negatives)), np.ones(len(self.scores_positives))))

        # Trier les scores et labels
        sorted_indices = np.argsort(scores)
        scores = scores[sorted_indices]
        labels = labels[sorted_indices]

        # Initialiser TPR et FPR
        tpr = []
        fpr = []
        n_pos = np.sum(labels)
        n_neg = len(labels) - n_pos

        tp = n_pos
        fp = n_neg

        # Calculer TPR et FPR pour chaque seuil
        for i in range(len(scores)):
            if labels[i] == 1:  # True positive
                tp -= 1
            else:  # False positive
                fp -= 1
            tpr.append(tp / n_pos)
            fpr.append(fp / n_neg)

        tpr = np.array(tpr)
        fpr = np.array(fpr)

        # Calculer l'AUROC
        auroc = -np.trapz(tpr, fpr)

        # Tracer la courbe ROC
        plt.figure()
        plt.plot(fpr, tpr, label=f"ROC Curve (AUROC = {auroc:.4f})")
        plt.plot([0, 1], [0, 1], 'k--', label="Random Classifier")
        plt.xlabel("False Positive Rate (FPR)")
        plt.ylabel("True Positive Rate (TPR)")
        plt.title("Receiver Operating Characteristic (ROC) Curve")
        plt.legend(loc="lower right")
        plt.grid()
        plt.show()

        return auroc
    

class LogitBasedScorer:
    @staticmethod
    def mls(logits):
        scores = - torch.max(torch.tensor(logits), dim=1)[0]
        return scores.cpu().numpy()

    @staticmethod
    def msp(logits):
        scores = torch.softmax(torch.tensor(logits), dim=1)
        scores = torch.max(scores, dim=1)[0]
        return scores.cpu().numpy()

    @staticmethod
    def energy(logits):
        scores = - torch.logsumexp(torch.tensor(logits), dim=1)
        return scores.cpu().numpy()

    @staticmethod
    def entropy(logits):
        probs = torch.softmax(torch.tensor(logits), dim=1)
        scores = - torch.sum(probs * torch.log(probs + 1e-10), dim=1)
        return scores.cpu().numpy()
    

class DKNN:
    def __init__(self, k=50, batch_size=256):
        self.k = k
        self.batch_size = batch_size
        self.fit_features = None

    def _l2_normalization(self, feat):
        norms = torch.norm(feat, p=2, dim=1, keepdim=True) + 1e-10  # Avoid division by zero
        return feat / norms

    def fit(self, fit_dataset):
        self.fit_features = self._l2_normalization(fit_dataset)

    def compute_scores(self, test_features):
        test_features = self._l2_normalization(test_features)
        scores = []

        # Process test features in batches
        for i in range(0, test_features.size(0), self.batch_size):
            batch = test_features[i:i + self.batch_size]
            # Compute pairwise distances for the batch
            distances = torch.cdist(batch, self.fit_features, p=2)  # (batch_size, num_fit_samples)
            # Sort distances and extract the k-th nearest
            sorted_distances, _ = torch.sort(distances, dim=1)
            scores.append(sorted_distances[:, self.k - 1])  # k-th nearest distance

        # Concatenate scores from all batches
        return torch.cat(scores, dim=0).cpu().numpy()    


class AnomalyDetector:
    def __init__(self, model, k=10):
        """
        Détecteur d'anomalies avec MLS, MSP, Energy, Entropy, DKNN et Mahalanobis.
        :param model: Modèles construits.
        :param k: Nombre de voisins pour DKNN.
        """
        pass

    def fit(self, features):
        """
        Entraîne le détecteur avec les caractéristiques fournies.
        :param features: Matrice de caractéristiques (films valides uniquement).
        """
        pass

    def predict(self, X): 
        pass 