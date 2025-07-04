
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.transforms import Normalize

class PerceptualLoss(nn.Module):
    """
    Calcola la Perceptual Loss utilizzando un modello VGG19 pre-addestrato.

    Questa loss confronta le feature map di alto livello estratte da un
    modello VGG, invece di confrontare i pixel direttamente. Questo porta
    a immagini generate che sono percettivamente più simili a quelle reali.
    """
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        
        # Carica il modello VGG19 pre-addestrato su ImageNet
        vgg = models.vgg19(pretrained=True).features
        
        # Selezioniamo un layer intermedio per estrarre le feature.
        # Il layer 35 corrisponde a 'conv5_4' in VGG19, che cattura
        # feature semantiche complesse.
        self.feature_extractor = nn.Sequential(*list(vgg.children())[:36]).eval()
        
        # Congeliamo i pesi del VGG, non vogliamo addestrarlo
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
            
        # Normalizzazione richiesta da VGG (media e std di ImageNet)
        self.normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # Usiamo la L1 Loss per confrontare le feature map
        self.criterion = nn.L1Loss()

    def forward(self, generated_image, real_image):
        """
        Args:
            generated_image (torch.Tensor): Immagine generata dal modello.
            real_image (torch.Tensor): Immagine reale dal dataset.

        Returns:
            torch.Tensor: Il valore della perceptual loss.
        """
        # Normalizza le immagini prima di darle in input a VGG
        norm_generated = self.normalize(generated_image)
        norm_real = self.normalize(real_image)
        
        # Estrai le feature map
        features_generated = self.feature_extractor(norm_generated)
        features_real = self.feature_extractor(norm_real)
        
        # Calcola la loss tra le feature map
        loss = self.criterion(features_generated, features_real)
        
        return loss
