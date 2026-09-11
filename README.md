# Plant Disease Classifier & Remedy
A full-stack agricultural AI application built with PyTorch, FastAPI, Streamlit, and Google Gemini Flash to detect plant leaf diseases and recommend actionable remedies.

---

## Model Weights & External Hosting
Due to GitHub's 100MB file size limit, large pre-trained ConvNeXt model weights (.pth) are excluded from version control. They are hosted externally on the Hugging Face Hub.

Hugging Face Repository: Harish39/plant_disease_classifier

Required File:

+ **plant_pre_trained_model.pth** - https://huggingface.co/Harish39/plant_disease_classifier/resolve/main/plant_pre_trained_model.pth

To set up the models locally, download them from the Hugging Face repository and place them inside the backend/models/ directory before starting the application.

---

## Getting Started
### 1. Clone the Repository
```git clone https://github.com/Harishkumar39/Plant-Disease-Classifier-and-Remedy.git```  
```cd Plant-Disease-Classifier-and-Remedy```  

### 2. Create and Activate Virtual Environment
```python -m venv .venv```  

On Windows:  
```.venv\Scripts\activate```

### 3. Configure Environment Variables
Create a ***.env*** file in the root directory and add your API keys:  
```GEMINI_API_KEY=your_gemini_api_key_here```

### 4. Run the Application
**Start FastAPI Backend:**  
```uvicorn backend.server:app --port 8000 --reload```  

**Start Streamlit Frontend:**  
```streamlit run classification_page.py```

---

## Model Training Results

* **Hardware**: CUDA (GPU)
* **Training Time**: 2,755.04 seconds (~45.9 minutes) for 5 epochs (On Kaggle Notebook T4 GPU)
* **Final Performance Metrics**:
  * **Train Loss**: 0.1604
  * **Train Accuracy**: 94.42%
  * **Test Loss**: 0.1703
  * **Test Accuracy**: 94.03% *(Peak test accuracy reached 94.09% at Epoch 3)*
 
 <img width="1189" height="490" alt="download" src="https://github.com/user-attachments/assets/6a5149a6-02c1-4c1c-b45d-c3fbf060e05a" />

