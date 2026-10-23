import torch
from torchvision import models, transforms
from PIL import Image
import urllib.request
import json

# 1. Download the "Labels" (the 1,000 categories the model knows)
url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
class_names = urllib.request.urlopen(url).read().decode("utf-8").splitlines()

# 2. Load the Pre-Trained Monster Model (ResNet-18)
# 'weights=models.ResNet18_Weights.DEFAULT' means "give me the version that already knows everything"
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.eval() # Put the model in "testing/see the world" mode

# 3. Prepare the Image (The model expects a specific format)
preprocess = transforms.Compose([
    transforms.Resize(256),      # Resize to 256 pixels
    transforms.CenterCrop(224),  # Crop the middle 224x224
    transforms.ToTensor(),       # Turn into math
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Color normalization
])

# Load your image (CHANGE THIS to your image file name!)
img_path = "my-gal.jpeg" 
img = Image.open(img_path).convert("RGB")
img_tensor = preprocess(img)
img_tensor = img_tensor.unsqueeze(0) # Add the "batch" dimension (remember this from earlier?)

# 4. Ask the model what it sees
with torch.no_grad():
    output = model(img_tensor)

# 5. Translate the math back to human words
probabilities = torch.nn.functional.softmax(output[0], dim=0)
top5_prob, top5_catid = torch.topk(probabilities, 5)

print(f"--- What the AI thinks your image is ---")
for i in range(top5_prob.size(0)):
    label = class_names[top5_catid[i]]
    score = top5_prob[i].item() * 100
    print(f"{label}: {score:.2f}% confidence")