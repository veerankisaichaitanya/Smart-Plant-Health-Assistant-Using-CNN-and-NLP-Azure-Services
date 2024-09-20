from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from tensorflow.keras.models import load_model
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
from gtts import gTTS
import pandas as pd
from io import BytesIO
from PIL import Image
import numpy as np
import os

# Initialize FastAPI
app = FastAPI()

# Load the CNN model for plant disease prediction
cnn_model = load_model('plant_disease_prediction_model.h5')

# Load the GPT-2 model and tokenizer
model_name = 'gpt2'
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Create the text generation pipeline
generator = pipeline('text-generation', model=model, tokenizer=tokenizer)

# Load the question-answering model
qa_pipeline = pipeline('question-answering', model='distilbert-base-uncased-distilled-squad', 
                       tokenizer='distilbert-base-uncased-distilled-squad')

# Load disease data from CSV
def load_disease_data(csv_file):
    df = pd.read_csv(csv_file)
    disease_data = {}
    for _, row in df.iterrows():
        disease_data[row['disease_name']] = {
            "symptoms": row['symptoms'],
            "causes": row['causes'],
            "treatments": row['treatments']
        }
    return disease_data

# Load your CSV file with plant disease data
disease_data = load_disease_data('sam.csv')

# Preprocess image for model prediction
def preprocess_image(image: Image.Image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Generate description using text generation model
def generate_description(disease_name):
    prompt = f"Provide a detailed description of the plant disease '{disease_name}', including symptoms, causes, and treatment methods."
    description = generator(prompt, max_length=200, num_return_sequences=1, truncation=True)
    return description[0]['generated_text']

# Convert text to speech
def text_to_speech(text, filename='output.mp3'):
    tts = gTTS(text=text, lang='en', slow=False)
    tts.save(filename)
    return filename

# Answer questions using the QA model
def answer_question(disease_name, question, data):
    context = data.get(disease_name, {})
    if not context:
        return "Disease information not found."

    context_text = (
        f"The symptoms of {disease_name} are: {context.get('symptoms', 'Not available')}.\n"
        f"The causes of {disease_name} include: {context.get('causes', 'Not available')}.\n"
        f"The treatments for {disease_name} are: {context.get('treatments', 'Not available')}."
    )
    result = qa_pipeline(question=question, context=context_text)
    return result['answer']

# API route to upload an image and get the predicted disease
@app.post("/predict-disease/")
async def predict_disease(image: UploadFile = File(...)):
    try:
        img = Image.open(BytesIO(await image.read()))
        img = preprocess_image(img)
        
        # Predict using the CNN model
        prediction = cnn_model.predict(img)
        predicted_index = np.argmax(prediction)
        
        # Example disease names list
        disease_names = ['Apple Scab', 'Black Rot', 'Cedar Apple Rust', 'Healthy Apple', 'Healthy Blueberry', 'Powdery Mildew Cherry', 
'Healthy Cherry', 'Cercospora Leaf Spot Corn', 'Common Rust Corn', 'Northern Leaf Blight Corn', 'Healthy Corn', 'Black Rot Grape', 
'Esca Grape', 'Leaf Blight Grape', 'Healthy Grape', 'Haunglongbing Orange', 'Bacterial Spot Peach', 'Healthy Peach', 'Bacterial Spot Pepper',
 'Healthy Pepper', 'Early Blight Potato', 'Late Blight Potato', 'Healthy Potato', 'Healthy Raspberry', 'Healthy Soybean', 'Powdery Mildew Squash',
   'Leaf Scorch Strawberry', 'Healthy Strawberry', 'Bacterial Spot Tomato', 'Early Blight Tomato', 'Late Blight Tomato', 'Leaf Mold Tomato', 
   'Septoria Leaf Spot Tomato', 'Spider Mites Tomato', 'Target Spot Tomato', 'Tomato Yellow Leaf Curl Virus', 'Tomato Mosaic Virus', 'Healthy Tomato']
        # ["disease1", "test_apple_black", "disease3"]  # Replace with actual names
        predicted_disease = disease_names[predicted_index]
        
        return {"predicted_disease": predicted_disease}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# API route to generate disease description
@app.get("/generate-description/{disease_name}")
def get_disease_description(disease_name: str):
    print("Requested disease name:", disease_name)
    print("Available disease names:", list(disease_data.keys()))
    
    if disease_name not in disease_data:
        raise HTTPException(status_code=404, detail="Disease not found")
    
    description = generate_description(disease_name)
    return {"description": description}


# API route to answer a question related to a disease
@app.get("/qa/")
def question_answer(disease_name: str, question: str):
    if disease_name not in disease_data:
        raise HTTPException(status_code=404, detail="Disease not found")
    answer = answer_question(disease_name, question, disease_data)
    return {"answer": answer}

# API route to get text-to-speech based on disease description from CSV
@app.get("/text-to-speech/{disease_name}")
def get_text_to_speech(disease_name: str):
    try:
        # Check if the disease exists in the CSV data
        if disease_name not in disease_data:
            raise HTTPException(status_code=404, detail="Disease not found in the dataset")

        # Fetch the disease description from CSV data
        disease_info = disease_data[disease_name]
        description = (
            f"Symptoms: {disease_info['symptoms']}.\n"
            f"Causes: {disease_info['causes']}.\n"
            f"Treatments: {disease_info['treatments']}."
        )

        # Convert the fetched description to speech
        filename = text_to_speech(description)

        # Return the speech file as a response
        return FileResponse(filename, media_type='audio/mp3', filename=os.path.basename(filename))
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
