
# 🌿 Plant Disease Detection using CNN

This project uses a **Convolutional Neural Network (CNN)** to classify plant leaf images into healthy or diseased categories. It is built using the **PlantVillage Dataset** and trained using **PyTorch**. The model aims to aid in early disease detection, ensuring better crop health and yield.

## 📁 Dataset Structure

The dataset contains images of healthy and diseased leaves from various plants like Tomato, Potato, and Bell Pepper. Categories include:

- Tomato_Bacterial_spot
- Tomato_Early_blight
- Tomato_Late_blight
- Tomato_Leaf_Mold
- Tomato_Septoria_leaf_spot
- Tomato_Spider_mites_Two_spotted_spider_mite
- Tomato__Target_Spot
- Tomato__Tomato_YellowLeaf__Curl_Virus
- Tomato__Tomato_mosaic_virus
- Tomato_healthy
- Potato___Early_blight
- Potato___Late_blight
- Potato___healthy
- Pepper__bell___Bacterial_spot
- Pepper__bell___healthy

## 🚀 Project Workflow

1. **Image Preprocessing**  
   - Resizing images to 128x128  
   - Normalization using `transforms.Normalize`  
   - Augmentation with RandomRotation and RandomHorizontalFlip

2. **Dataset Loading**  
   - Train/Validation split using `torchvision.datasets.ImageFolder`
   - `DataLoader` used for batching

3. **Model Architecture**

Custom CNN architecture:
Sequential(
  (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (1): ReLU()
  (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (3): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (4): ReLU()
  (5): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (6): Flatten(start_dim=1, end_dim=-1)
  (7): Linear(in_features=65536, out_features=256, bias=True)
  (8): ReLU()
  (9): Dropout(p=0.5, inplace=False)
  (10): Linear(in_features=256, out_features=15, bias=True)

- **Loss Function**: CrossEntropyLoss
- **Optimizer**: Adam (lr=0.001)
- **Batch Size**: 32
- **Epochs**: 5
- **Framework**: PyTorch

4. **Training**  
   - Optimizer: Adam  
   - Loss Function: CrossEntropyLoss  
   - Epochs: 10 (can be tuned)  
   - Model training and validation loops

<img width="370" height="105" alt="image" src="https://github.com/user-attachments/assets/ca055e45-30cf-4e61-bf4c-2a21cc08d2bb" />

5. **Evaluation**  

<img width="767" height="438" alt="image" src="https://github.com/user-attachments/assets/01a1f188-4b7b-46d9-ab4b-03ab43cc70fb" />


<img width="732" height="415" alt="image" src="https://github.com/user-attachments/assets/0d32b2ce-dc4d-4c25-9871-c9761b25e7db" />

   - Confusion Matrix visualization

<img width="716" height="647" alt="image" src="https://github.com/user-attachments/assets/06b5e8fe-55c3-4510-b838-75d0ae74adb0" />

   - Sample predictions with image display

<img width="399" height="362" alt="image" src="https://github.com/user-attachments/assets/5ce4c77c-099f-439e-bcac-f9fa555f3806" />

## **Conclusion**
The CNN-based model effectively classifies plant leaf diseases with high accuracy using the PlantVillage dataset. It offers a reliable solution for early disease detection in agriculture, aiding timely intervention.
