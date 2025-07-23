from PIL import Image
import torchvision.transforms as transforms
import torch
import torchvision.models as models

# Load pretrained MobileNet v2 model
model = models.mobilenet_v2(pretrained=True)
model.eval()

# Load ImageNet class labels
with open('imagenet_classes.txt') as f:
    imagenet_classes = [line.strip() for line in f.readlines()]

def analyze_image(image_path):
    input_image = Image.open(image_path).convert('RGB')

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        ),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)  # create batch dimension

    with torch.no_grad():
        output = model(input_batch)

    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    # Get top 3 classes
    top3_prob, top3_catid = torch.topk(probabilities, 3)

    result = []
    for i in range(top3_prob.size(0)):
        result.append((imagenet_classes[top3_catid[i]], top3_prob[i].item()))

    return result
