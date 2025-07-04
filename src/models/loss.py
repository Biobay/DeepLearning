import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.transforms import Normalize

class PerceptualLoss(nn.Module):
    """
    Calcola la VGG-based Perceptual Loss.

    Utilizza le feature map estratte da un modello VGG19 pre-addestrato per
    confrontare l'immagine generata e quella reale. La loss è la Mean Squared Error
    tra le attivazioni di specifici layer intermedi.
    """
    def __init__(self, device="cpu"):
        super(PerceptualLoss, self).__init__()
        # Carichiamo il modello VGG19 pre-addestrato
        vgg19 = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features.to(device)
        
        # Selezioniamo i layer fino al blocco conv4_4 (layer 35)
        # Questo è un punto comune per la perceptual loss, ma può essere modificato
        self.features_extractor = nn.Sequential(*list(vgg19.children())[:36]).eval()
        
        # Congeliamo i pesi del VGG, non dobbiamo addestrarlo
        for param in self.features_extractor.parameters():
            param.requires_grad = False
            
        # La loss tra le feature map sarà una MSE
        self.criterion = nn.MSELoss()

        # Normalizzazione richiesta dai modelli pre-addestrati su ImageNet
        self.normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self, generated_img, real_img):
        """
        Calcola la perceptual loss.

        Args:
            generated_img (torch.Tensor): Immagine generata dal modello.
            real_img (torch.Tensor): Immagine reale dal dataset.

        Returns:
            torch.Tensor: Il valore della perceptual loss.
        """
        # Normalizziamo le immagini prima di darle in input al VGG
        # Assumiamo che le immagini in input siano in range [-1, 1] o [0, 1]
        # Le portiamo a [0, 1] se necessario
        if generated_img.min() < 0:
            generated_img = (generated_img + 1) / 2
        if real_img.min() < 0:
            real_img = (real_img + 1) / 2
            
        norm_generated = self.normalize(generated_img)
        norm_real = self.normalize(real_img)

        # Estraiamo le feature
        generated_features = self.features_extractor(norm_generated)
        real_features = self.features_extractor(norm_real)
        
        # Calcoliamo la loss
        loss = self.criterion(generated_features, real_features)
        
        return loss

class CombinedLoss(nn.Module):
    """
    Combina Perceptual Loss e L1 Loss per bilanciare struttura e colori.
    """
    def __init__(self, l1_weight=1.0, device="cpu"):
        super(CombinedLoss, self).__init__()
        self.perceptual_loss = PerceptualLoss(device=device)
        self.l1_loss = nn.L1Loss()
        self.l1_weight = l1_weight

    def forward(self, generated_img, real_img):
        # Calcola le due componenti della loss
        perceptual = self.perceptual_loss(generated_img, real_img)
        l1 = self.l1_loss(generated_img, real_img)
        
        # Combina le loss con il peso specificato
        combined_loss = perceptual + self.l1_weight * l1
        
        return combined_loss
