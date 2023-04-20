Face Recognition using PCA and K-Nearest Neighbors Classifier
-------------------------------------------------------------

This Python code demonstrates face recognition using Principal Component Analysis (PCA) and K-Nearest Neighbors (KNN) classifier. The code reads face images from a zip file, performs PCA to extract eigenfaces, and then trains a KNN classifier on the eigenfaces for face recognition.

### Libraries Used

-   `zipfile`: Python built-in library for working with zip files.
-   `cv2`: OpenCV library for image processing and computer vision tasks.
-   `numpy`: NumPy library for numerical computing in Python.
-   `matplotlib`: Matplotlib library for data visualization in Python.
-   `sklearn`: Scikit-learn library for machine learning tasks, including PCA, model selection, and evaluation.

### Code Flow

1.  Reading Face Images from Zip File:

    -   The code uses the `zipfile` library to open a zip file called "attface.zip".
    -   It iterates through the files in the zip file and reads the face images in PGM format using OpenCV's `imdecode()` function.
    -   The face images are stored in a dictionary called `faces`, where the keys are the filenames and the values are the image data in grayscale format.
2.  Showing Sample Faces:

    -   The code uses `matplotlib` to display 16 sample faces from the loaded images using subplots.
    -   The images are displayed in a 4x4 grid with shared x-axis and y-axis, and using a grayscale color map.
3.  Data Preparation:

    -   The code prepares the face image data for PCA and classification by flattening the images into 1D arrays and storing them in a NumPy array called `facematrix`.
    -   The corresponding class labels (i.e., the names of the individuals) are stored in a list called `facelabel`.
4.  Performing PCA:

    -   The code uses Scikit-learn's `PCA` class to perform Principal Component Analysis on the `facematrix` data.
    -   The number of components to keep (i.e., the number of eigenfaces) is set to 16.
    -   The mean face and eigenfaces are obtained from the PCA model.
5.  Displaying Eigenfaces:

    -   The code uses `matplotlib` to display the first 16 eigenfaces obtained from PCA.
    -   The eigenfaces are reshaped back into 2D images using the original face image shape.
6.  Splitting Data into Training and Test Sets:

    -   The code uses Scikit-learn's `train_test_split()` function to split the `facematrix` data into training and test sets.
    -   The training set is used for training the KNN classifier, and the test set is used for evaluating the classification performance.
7.  Training KNN Classifier:

    -   The code creates an instance of the KNN classifier from Scikit-learn's `KNeighborsClassifier` class, with the number of neighbors set to 3.
    -   The training data, obtained by projecting the original face image data onto the eigenfaces, is used to train the KNN classifier using the `fit()` method.
8.  Testing and Evaluation:

    -   The code projects the test data onto the eigenfaces using the PCA model.
    -   The KNN classifier is then used to predict the class labels for the test data using the `predict()` method.
    -   The accuracy of the classifier is calculated using Scikit-learn's `accuracy_score()` function, and a classification report is printed using Scikit-learn's `classification_report()` function.

### Usage

-   Ensure that the "attface.zip" file containing the face images is available in the same directory as the codefile.

-   Make sure that the required libraries (`zipfile`, `cv2`, `numpy`, `matplotlib`, and `sklearn`) are installed in your Python environment.
-   Run the code in a Python environment or IDE of your choice.
-   The code will load the face images from the zip file, display sample faces, perform PCA to extract eigenfaces, train a KNN classifier, and evaluate the classification performance.
-   The accuracy of the classifier and the classification report will be printed on the console, indicating the performance of the face recognition system.

### Notes

-   The accuracy of the face recognition system may vary depending on the quality of the face images, the number of training samples, and the choice of hyperparameters (e.g., number of components in PCA, number of neighbors in KNN).
-   You can experiment with different hyperparameter values to achieve better performance or try other machine learning algorithms for face recognition.
-   It's important to have a sufficiently large and diverse dataset of face images for training and testing the face recognition system to achieve better accuracy.
-   Ensure that you have the necessary permissions to use the face images for face recognition purposes and comply with any legal or ethical requirements.

### Conclusion

This code provides a basic implementation of face recognition using PCA and KNN classifier. It can serve as a starting point for developing more advanced face recognition systems with higher accuracy and robustness. Remember to use it responsibly and follow all relevant laws and regulations regarding privacy and data usage.
