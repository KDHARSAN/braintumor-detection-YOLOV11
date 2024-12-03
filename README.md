# braintumor-detection-YOLOV11
 This repository features a YOLOv8 model for tumor detection in medical images. Trained on a custom dataset for 50 epochs with 640x640 images, it delivers precise results with minimal loss. The model supports seamless training, evaluation, and visualization of predictions, offering a robust tool for advancing medical imaging diagnostics



# Tumor Detection Using YOLOv8  

This repository implements a YOLOv8 model for tumor detection, designed for efficient and accurate identification of tumor regions in medical images.  

## Project Overview  

The YOLOv8 model is trained on a custom dataset using a tailored configuration to optimize detection performance. The training process spans 50 epochs with an input image size of 640x640, ensuring a balance between speed and precision.  

### Key Features  
- **Custom Dataset**: Utilizes a dataset configured via a `data.yaml` file for seamless integration.  
- **Model Training**: The model is trained using the YOLO framework, leveraging advanced optimization techniques to improve detection accuracy.  
- **Evaluation**: The trained model is tested on a separate dataset to validate performance and generate detection outputs with precision bounding boxes.  
- **Performance Metrics**: Achieves robust results with a focus on minimizing training loss and maximizing detection accuracy.  

## How to Use  

1. **Train the Model**  
   Run the following command to start training:  
   ```bash  
   !yolo detect train data=/path/to/data.yaml epochs=50 imgsz=640  
   ```  

2. **Evaluate the Model**  
   Load and evaluate the trained model on test images:  
   ```python  
   from ultralytics import YOLO  

   model = YOLO("/path/to/weights/best.pt")  
   results = model("/path/to/test/image.jpg", save=True)  
   ```  

3. **Results Visualization**  
   The model outputs annotated images showing detected tumor regions, enabling visual analysis.  

## Future Work  
This project sets the foundation for enhancing tumor detection accuracy. Future improvements could include fine-tuning the model on larger datasets, integrating additional metrics for evaluation, and deploying the system for real-time detection applications.  


