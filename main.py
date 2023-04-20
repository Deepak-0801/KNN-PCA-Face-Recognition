import zipfile
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Read face image from zip file on the fly
faces = {}
with zipfile.ZipFile("attface.zip") as facezip:
    for filename in facezip.namelist():
        if not filename.endswith(".pgm"):
            continue # not a face picture
        with facezip.open(filename) as image:
            img_data = np.frombuffer(image.read(), np.uint8)
            img = cv2.imdecode(img_data, cv2.IMREAD_GRAYSCALE)
            faces[filename] = img

# Show sample faces using matplotlib
fig, axes = plt.subplots(4, 4, sharex=True, sharey=True, figsize=(8, 10))
faceimages = list(faces.values())[-16:] # take last 16 images
for i in range(16):
    axes[i%4][i//4].imshow(faceimages[i], cmap="gray")
print("Showing sample faces")
plt.show()

# Print some details
faceshape = next(iter(faces.values())).shape
print("Face image shape:", faceshape)

classes = set(filename.split("/")[0] for filename in faces.keys())
print("Number of classes:", len(classes))
print("Number of images:", len(faces))

# Prepare the data for PCA and classification
facematrix = []
facelabel = []
for key, val in faces.items():
    facematrix.append(val.flatten())
    facelabel.append(key.split("/")[0])

# Create a NxM matrix with N images and M pixels per image
facematrix = np.array(facematrix)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(facematrix, facelabel, test_size=0.2, random_state=42)

# Apply PCA and take first K principal components as eigenfaces
n_components = 16
pca = PCA(n_components=n_components).fit(X_train)
mean_face = pca.mean_
eigenfaces = pca.components_

# Show the first 16 eigenfaces
fig, axes = plt.subplots(4, 4, sharex=True, sharey=True, figsize=(8, 10))
for i in range(16):
    axes[i%4][i//4].imshow(eigenfaces[i].reshape(faceshape), cmap="gray")
print("Showing the eigenfaces")

# Project the training data onto the eigenfaces
X_train_pca = pca.transform(X_train)

# Project the test data onto the eigenfaces
X_test_pca = pca.transform(X_test)

# Train a KNN classifier on the training data
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_pca, y_train)

# Predict labels for the test data
y_pred = knn.predict(X_test_pca)

# Calculate accuracy and print classification report
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))
print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=1))
