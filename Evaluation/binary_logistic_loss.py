# used in contrastive learning 
def binary_logistic_loss(outputs: FloatTensor, positive: bool):
    """Utility function to wrap ``torch.SoftMarginLoss``."""
    loss = nn.SoftMarginLoss()
    if positive:
        return loss(outputs, torch.ones(len(outputs)).to(outputs.device))
    else:
        return loss(outputs, -torch.ones(len(outputs)).to(outputs.device))
