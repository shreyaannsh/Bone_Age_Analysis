# Bone-Age-Analysis

Bone Age Prediction using Deep Learning 🦴🧠
This repository contains a deep learning-based approach for predicting bone age from hand X-ray images. It was developed as part of a machine learning project and uses TensorFlow and Keras to train a CNN (Convolutional Neural Network) model on the RSNA Pediatric Bone Age Challenge Dataset.

📂 Project Structure
bash
Copy
Edit
bone-age-prediction/
├── boneagemlhackathon (1).ipynb   # Main notebook containing data preprocessing, model training, and evaluation
├── README.md                      # This file
└── /images                        # Optional: folder for saving training history plots or sample outputs
📌 Features
Reads metadata from CSV file (image ID and corresponding bone age)

Preprocesses and normalizes grayscale hand X-ray images

Builds a CNN model using Keras with Conv2D, MaxPooling, and Dense layers

Trains the model on image data with regression output (predicting bone age in months)

Evaluates the model using Mean Absolute Error (MAE)

🧪 Requirements
Install the following packages using pip install -r requirements.txt:

bash
Copy
Edit
tensorflow
numpy
matplotlib
pandas
scikit-learn
opencv-python
Or install them manually:

bash
Copy
Edit
pip install tensorflow numpy matplotlib pandas scikit-learn opencv-python
🚀 How to Run
Clone the repository:

bash
Copy
Edit
git clone https://github.com/yourusername/bone-age-prediction.git
cd bone-age-prediction
Place the dataset in the correct path:

Ensure you have the image dataset (boneage-training-dataset) and the CSV file with image metadata.

Open the notebook and run all cells:

boneagemlhackathon (1).ipynb contains the full pipeline from data loading to evaluation.

(Optional) Save the trained model for later inference.

📊 Results
The model is evaluated using MAE (Mean Absolute Error).

Visualization of training/validation loss is included to monitor model performance.

🛠️ Future Improvements
Hyperparameter tuning using Optuna or KerasTuner

Use data augmentation to improve model generalization

Experiment with different architectures (e.g., ResNet, EfficientNet)

Use pre-trained models via transfer learning

🤝 Contributing
Feel free to fork this repository and improve the model, or submit a pull request if you have something useful to add!

📜 License
This project is licensed under the MIT License.
