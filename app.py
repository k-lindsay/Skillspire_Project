import torch
import gradio as gr
from torchvision import transforms
from PIL import Image
from timeit import default_timer as timer
from model import boat_detection_model

# Classes
class_names = ['Cargo', 'Military', 'Carrier', 'Cruise', 'Tanker']

# Load Model
model = boat_detection_model(input=3, hiddenwork=16, output=5)
model.load_state_dict(torch.load("02NOV2026_final_model.pth", map_location=torch.device('cpu')))
model.eval()

# Transform pipelines matching dataset setup
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict(img):
    start_time = timer()
    # Transform and add batch dimension
    img_tensor = transform(img).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        
    # Prepare response dictionary for labels
    predictions = {class_names[i]: float(probs[i]) for i in range(len(class_names))}
    inference_time = round(timer() - start_time, 4)
    return predictions, inference_time

# Gradio interface configuration
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Label(num_top_classes=5, label="Probabilities"),
        gr.Number(label="Prediction Time (s)")
    ],
    title="🚢 Ship Category Classifier 🌊",
    description="Upload an image of a marine vessel to classify it as Cargo, Military, Carrier, Cruise, or Tanker!"
)

if __name__ == '__main__':
    demo.launch()
