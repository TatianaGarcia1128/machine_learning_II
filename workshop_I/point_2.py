#import modules
from PIL import Image
import numpy as np
from io import BytesIO
import requests
import matplotlib.pyplot as plt
import os

#Image processing function
def process_image(path: str):
    '''
    Save and plot processed image with the following features:
        * Grayscale
        * 256x256 pixels

            Args:
              path (str): path where the image is saved in quotes.

            Returns:
              Plot of original image
              Plot of proccessed image
              Image saved in the folder

            Example:
              >>> process_image("path of image")
    '''

    #Image reading
    personal_photo = Image.open(path)

    #Convert image to grayscale
    grayscale_image = personal_photo.convert('L')

    #Resize image to 256x256
    resized_image = grayscale_image.resize((256, 256))

    #Save the processed image
    resized_image.save('tatiana_garcia.jpg')

    #Show original image
    plt.imshow(np.asarray(personal_photo), cmap='gray')
    plt.title('Original image')
    plt.axis('off')
    plt.show()

    #Show the image in grayscale and resized
    plt.imshow(np.asarray(resized_image), cmap='gray')
    plt.title('Proccessed image')
    plt.axis('off')
    plt.show()

#Function to calculate the average face of the cohort.
def calculate_average_face_cohort(path_cohort: str):
    '''
    Calculate and plot the average face of the cohort.

            Args:
              path (str): path where the cohort images are saved

            Returns:
              The average face of the cohort.

            Example:
              >>> calculate_average_face_cohort("path of images")
    '''
        
    #Get the list of image file names in the directory
    list_images = [file for file in os.listdir(path_cohort) if file.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]
    
    #Get the dimensions of the first image
    height, wide = Image.open(os.path.join(path_cohort, list_images[0])).size

    #Initialize a matrix to store the sum of all images
    sum_images = np.zeros((height, wide), dtype=np.float32)


    for file in list_images:
          # Obtener el nombre del archivo y su extensión
        file_name, file_extension = os.path.splitext(file)
        
        # Si la extensión es .png, cambiarla a .jpg
        if file_extension.lower() == '.png':
            new_file_name = file_name + '.jpg'
            os.rename(os.path.join(path_cohort, file), os.path.join(path_cohort, new_file_name))
        else:
            new_file_name = file
            
        #Image reading
        image_cohort = Image.open(os.path.join(path_cohort, new_file_name)).convert('L').resize((256, 256))
        #Convert image to a numpy array 
        imagen_cohort_array = np.array(image_cohort)
        #Convert array elements to float32
        sum_images += imagen_cohort_array.astype(np.float32)

    #Calculate the average of all images
    average_images = sum_images / len(list_images)

    return average_images


#Function to calculate and plot the average face of the cohort.
def plot_average_face_cohort(path_cohort: str):
    '''
    Calculate and plot the average face of the cohort.

            Args:
              path_cohort (str): path where the cohort images are saved

            Returns:
              The average face of the cohort.

            Example:
              >>> calculate_average_face_cohort("path of images")
    '''
        
    average_images = calculate_average_face_cohort(path_cohort)

    #Convert average image to uint8 data type
    average_uint8 = average_images.astype(np.uint8)

    #Show average image using Matplotlib
    plt.imshow(average_uint8, cmap='gray')
    plt.axis('off')
    plt.show()

#Function distant is your face from the average
def distance_face_from_average(path_face: str, path_faces: str):
    '''
    Calculates the cosine distance which measures the angle between two vectors and can be useful when you are interested in the directional 
    similarity between the vectors rather than their magnitude.
    The cosine distance between two vectors u and v are calculated as the cosine of the angle between them, and are defined as follows:

                d(u,v)=1 - (u . v) / ||u|| ||v||

    Where u⋅v is the dot product between the vectors u and v, and ||u|| y ||v|| are the Euclidean norms of the vectors u and v, respectively.

            Args:
              path_face (str): path where my processed photo is saved
              path_faces (str): path where the cohort images are saved

            Returns:
              The average face of the cohort.

            Example:
              >>> calculate_average_face_cohort("path of images")
    '''
        
    #My face features vector
    vector_my_face =  Image.open(os.path.join(path_face))
    vector_my_face_array = np.array(vector_my_face).astype(np.float32)

    #Average feature vector of faces
    vector_average_faces = calculate_average_face_cohort(path_faces)

    #Calculate cosine distance
    distance_cos = 1 - np.dot(vector_my_face_array, vector_average_faces) / (np.linalg.norm(vector_my_face_array) * np.linalg.norm(vector_average_faces))

    print("The cosine distance between your face and the average of the faces is: \n", distance_cos)



#Path definition
path_original_photo = "/Users/tatianagarcia/Library/Mobile Documents/com~apple~CloudDocs/Especialización ciencia de datos y analítica/Machine Learning 2/tatiana_garcia_original.jpg"
path_photos_cohort = "/Users/tatianagarcia/Library/Mobile Documents/com~apple~CloudDocs/Especialización ciencia de datos y analítica/Machine Learning 2/photos_group"
path_processed_photo ="/Users/tatianagarcia/Library/Mobile Documents/com~apple~CloudDocs/Especialización ciencia de datos y analítica/Machine Learning 2/machine_learning_II/tatiana_garcia.jpg"

#Use of functions
#process_image(path_original_photo)
plot_average_face_cohort(path_photos_cohort)
distance_face_from_average(path_processed_photo, path_photos_cohort)
