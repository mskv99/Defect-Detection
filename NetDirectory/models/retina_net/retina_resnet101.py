import torch
import torchvision
from torchvision.models.detection import RetinaNet
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.resnet import ResNet101_Weights

def create_model(num_classes = 5):
    backbone = torchvision.models.resnet101(weights=ResNet101_Weights)

    new_backbone = torch.nn.Sequential(*list(backbone.children())[:-2])
    print(*list(backbone.children())[:-2])


    new_backbone.out_channels = 2048

    anchor_generator = AnchorGenerator(
         sizes=((32, 64, 128, 256, 512),),
         aspect_ratios=((0.5, 1.0, 2.0),)
     )

     # put the pieces together inside a RetinaNet model
    model = RetinaNet(new_backbone,
                       num_classes=num_classes,
                       anchor_generator=anchor_generator)
    return model

if __name__ == '__main__':
    model = create_model(5)
    print(model)
    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")
