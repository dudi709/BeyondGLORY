import torch
import torch.nn as nn
import torch.nn.functional as F


class NCELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, score, label):
        """

        Args:
            score: (batch_size, candidate_num)
            label: (batch_size, candidate_num)

        Returns:

        """
        # (batch_size)
        result = F.log_softmax(score, dim=1)
        loss = F.nll_loss(result, label)

        return loss


class EnhancedNCELossRanking(nn.Module):
    def __init__(self, alpha=1.0, margin=1.0):
        super().__init__()
        self.alpha = alpha
        self.margin = margin

    def forward(self, scores, labels):
        """
        Args:
            scores: A tensor of shape (batch_size, candidate_num) containing the raw scores for each candidate.
            labels: A tensor of shape (batch_size, candidate_num) containing the labels, where the first candidate is positive and the rest are negative.

        Returns:
            The combined NCE and pairwise ranking loss.
        """
        # NCE loss component
        log_probs = F.log_softmax(scores, dim=1)
        nce_loss = F.nll_loss(log_probs, labels)

        # Pairwise ranking loss component
        positive_scores = scores[:, 0].unsqueeze(1)  # Get the scores of the positive samples and unsqueeze for broadcasting
        negative_scores = scores[:, 1:]  # Get the scores of the negative samples
        # Calculate the ranking loss for all positive-negative pairs
        ranking_loss = torch.sum(torch.clamp(self.margin - (positive_scores - negative_scores), min=0))

        # Combine the losses
        combined_loss = nce_loss + self.alpha * ranking_loss / (scores.size(0) * (scores.size(1) - 1))

        return combined_loss


class PairwiseRankingLoss(nn.Module):
    def __init__(self, alpha=1.0, margin=1.0):
        super().__init__()
        self.alpha = alpha
        self.margin = margin

    def forward(self, scores, labels):
        """
        Args:
            scores: A tensor of shape (batch_size, candidate_num) containing the raw scores for each candidate.
            labels: A tensor of shape (batch_size, candidate_num) containing the labels, where the first candidate is positive and the rest are negative.

        Returns:
            The combined NCE and pairwise ranking loss.
        """

        # Pairwise ranking loss component
        positive_scores = scores[:, 0].unsqueeze(1)  # Get the scores of the positive samples and unsqueeze for broadcasting
        negative_scores = scores[:, 1:]  # Get the scores of the negative samples
        # Calculate the ranking loss for all positive-negative pairs
        ranking_loss = torch.sum(torch.clamp(self.margin - (positive_scores - negative_scores), min=0))

        # Combine the losses
        combined_loss = self.alpha * ranking_loss / (scores.size(0) * (scores.size(1) - 1))

        return combined_loss