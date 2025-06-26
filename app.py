import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np

# Custom imports
from src.preprocessing.dataLoader_CelebA import get_partitioned_dataloaders
from src.ml.own_network import SiameseNetworkOwn
from src.ml.resNet18 import SiameseNetwork

# DEVICE SETUP
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model Loading with Thresholds
models_dict = {
    # Format: {"model_name": {"model": model, "threshold": threshold_value}}
    "MS_OWN_3": {"path": "models/MS_OWN_3.pth", "threshold": 0.5},
    "ARCFACE_OWN_3": {"path": "models/ARCFACE_OWN_3.pth", "threshold": 0.35}
}

def load_model(model_info):
    model_path = model_info["path"]
    if "RESNET" in model_path:
        model = SiameseNetwork(loss_type=model_path.split("_")[0].lower()).to(DEVICE)
    else:
        model = SiameseNetworkOwn(loss_type=model_path.split("_")[0].lower()).to(DEVICE)
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model

# Load all models
for model_name, model_info in models_dict.items():
    print(f"🔍 Loading: {model_name}")
    models_dict[model_name]["model"] = load_model(model_info)

def process_image(pil_img):
    # Convert to grayscale first
    if pil_img.mode != 'L':
        pil_img = pil_img.convert('L')
    transform = T.Compose([
        T.Resize([100, 100]),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5]),  # Grayscale normalization
    ])
    print(pil_img)
    return transform(pil_img).unsqueeze(0).to(DEVICE)

def get_metrics_for_models(models_dict, img1, img2):
    results = {}
    with torch.no_grad():
        for model_name, model_info in models_dict.items():
            model = model_info["model"]
            threshold = model_info["threshold"]
            
            # Get embeddings
            emb1 = model.forward_once(img1)
            emb2 = model.forward_once(img2)
            
            # Calculate cosine similarity
            emb1_np = emb1.cpu().numpy().flatten()
            emb2_np = emb2.cpu().numpy().flatten()
            similarity = np.dot(emb1_np, emb2_np) / (np.linalg.norm(emb1_np) * np.linalg.norm(emb2_np))
            
            # Determine if same person
            is_same = similarity > threshold
            
            results[model_name] = {
                "similarity": float(similarity),
                "is_same": bool(is_same),
                "threshold": float(threshold)
            }
    return results

# Cropping Window (unchanged)
class ImageCropper(tk.Toplevel):
    def __init__(self, master, image_path, callback, crop_size=(400, 400)):
        super().__init__(master)
        self.title("Crop Image")
        self.callback = callback
        self.original = Image.open(image_path).convert('RGB')
        self.tk_image = ImageTk.PhotoImage(self.original)
        self.canvas = tk.Canvas(self, width=self.original.width, height=self.original.height)
        self.canvas.pack()
        self.canvas.create_image(0, 0, image=self.tk_image, anchor="nw")
        self.rect = None
        self.start_x = self.start_y = None
        self.canvas.bind("<ButtonPress-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

    def on_click(self, event):
        self.start_x = event.x
        self.start_y = event.y
        if self.rect:
            self.canvas.delete(self.rect)
        self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, event.x, event.y, outline='red')

    def on_drag(self, event):
        self.canvas.coords(self.rect, self.start_x, self.start_y, event.x, event.y)

    def on_release(self, event):
        x0, y0, x1, y1 = self.canvas.coords(self.rect)
        cropped = self.original.crop((x0, y0, x1, y1))
        self.callback(cropped)
        self.destroy()

# Main App with enhanced results display
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Comparison App")
        self.img1 = None
        self.img2 = None
        self.tensor1 = None
        self.tensor2 = None

        # UI Layout
        button_frame = tk.Frame(root)
        button_frame.pack(pady=10)

        image_frame = tk.Frame(root)
        image_frame.pack()

        result_frame = tk.Frame(root)
        result_frame.pack(pady=10)

        # Buttons
        tk.Button(button_frame, text="Upload Image 1", command=self.upload_img1).grid(row=0, column=0, padx=10)
        tk.Button(button_frame, text="Upload Image 2", command=self.upload_img2).grid(row=0, column=1, padx=10)
        tk.Button(button_frame, text="Compare Images", command=self.compare_images).grid(row=0, column=2, padx=10)

        # Image Preview
        self.img1_label = tk.Label(image_frame, text="Image 1", width=150, height=150, bg="gray")
        self.img1_label.grid(row=0, column=0, padx=10)

        self.img2_label = tk.Label(image_frame, text="Image 2", width=150, height=150, bg="gray")
        self.img2_label.grid(row=0, column=1, padx=10)

        # Results
        self.result_box = tk.Text(result_frame, height=12, width=60)
        self.result_box.pack()

    def upload_img1(self):
        path = filedialog.askopenfilename()
        if path:
            ImageCropper(self.root, path, self.set_img1)

    def upload_img2(self):
        path = filedialog.askopenfilename()
        if path:
            ImageCropper(self.root, path, self.set_img2)

    def set_img1(self, pil_img):
        self.img1 = pil_img
        self.tensor1 = process_image(self.img1)
        self.display_image(self.img1, self.img1_label)

    def set_img2(self, pil_img):
        self.img2 = pil_img
        self.tensor2 = process_image(self.img2)
        self.display_image(self.img2, self.img2_label)

    def display_image(self, pil_img, label_widget):
        display_img = pil_img.resize((150, 150))
        tk_img = ImageTk.PhotoImage(display_img)
        label_widget.configure(image=tk_img)
        label_widget.image = tk_img  # keep reference

    def compare_images(self):
        if self.tensor1 is None or self.tensor2 is None:
            self.result_box.insert(tk.END, "Please upload and crop both images.\n")
            return
            
        metrics = get_metrics_for_models(models_dict, self.tensor1, self.tensor2)
        self.result_box.delete(1.0, tk.END)
        
        for model_name, result in metrics.items():
            similarity = result["similarity"]
            is_same = result["is_same"]
            threshold = result["threshold"]
            
            # Color coding
            color = "green" if is_same else "red"
            decision = "SAME PERSON" if is_same else "DIFFERENT PEOPLE"
            
            self.result_box.insert(tk.END, f"{model_name}:\n", "bold")
            self.result_box.insert(tk.END, f"  Similarity: {similarity:.4f}\n")
            self.result_box.insert(tk.END, f"  Threshold: {threshold:.4f}\n")
            self.result_box.insert(tk.END, f"  Decision: ", "bold")
            self.result_box.insert(tk.END, f"{decision}\n\n", color)
            
        # Configure text colors
        self.result_box.tag_config("bold", font=('Arial', 10, 'bold'))
        self.result_box.tag_config("green", foreground="green")
        self.result_box.tag_config("red", foreground="red")

# Run App
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()