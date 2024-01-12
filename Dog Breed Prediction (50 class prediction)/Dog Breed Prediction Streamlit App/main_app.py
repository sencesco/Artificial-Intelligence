#Library imports
import numpy as np
import streamlit as st
import cv2
from keras.models import load_model


#Loading the Model
model = load_model('dog_breed.h5')

#Name of Classes
CLASS_NAMES = ['pekinese', 'walker_hound', 'boxer', 'otterhound',
       'english_setter', 'dhole', 'toy_poodle', 'border_terrier',
       'norwegian_elkhound', 'shih-tzu', 'kuvasz', 'german_shepherd',
       'greater_swiss_mountain_dog', 'australian_terrier',
       'rhodesian_ridgeback', 'appenzeller', 'samoyed', 'border_collie',
       'entlebucher', 'collie', 'malamute', 'chihuahua', 'saluki',
       'komondor', 'bull_mastiff', 'bernese_mountain_dog', 'lhasa',
       'scotch_terrier', 'miniature_pinscher', 'brabancon_griffon',
       'toy_terrier', 'flat-coated_retriever',
       'soft-coated_wheaten_terrier', 'siberian_husky', 'briard',
       'chesapeake_bay_retriever', 'beagle', 'vizsla',
       'west_highland_white_terrier', 'kerry_blue_terrier', 'whippet',
       'japanese_spaniel', 'curly-coated_retriever', 'pembroke',
       'silky_terrier', 'sussex_spaniel', 'german_short-haired_pointer',
       'french_bulldog', 'english_springer', 'rottweiler']

#Setting Title of App
st.title("Dog Breed Prediction")
st.markdown("Upload an image of the dog")

#Uploading the dog image
dog_image = st.file_uploader("Choose an image...", type=["png","jpg","jpeg"])
submit = st.button('Predict')
#On predict button click
if submit:


    if dog_image is not None:

        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(dog_image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)



        # Displaying the image
        st.image(opencv_image, channels="BGR")
        #Resizing the image
        opencv_image = cv2.resize(opencv_image, (224,224))
        #Convert image to 4 Dimension
        opencv_image.shape = (1,224,224,3)
        #Make Prediction
        Y_pred = model.predict(opencv_image)

        st.title(str("The Dog Breed is "+CLASS_NAMES[np.argmax(Y_pred)]))
        
    else:
        st.write('Pleas upload image')
