import torch
import torchvision.models as models

# Replace this with your actual trained model
model = models.mobilenet_v2(pretrained=True)  
model.eval()

scripted_model = torch.jit.script(Models/mobilenetv2_fruits_scripted.pt)  # or use torch.jit.trace
scripted_model.save("Models/mobilenetv2_fruits_scripted(graph).pt")
