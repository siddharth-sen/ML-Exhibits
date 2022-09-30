import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
import os
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

read_file_train = os.path.join("..", "..", "data", "pca_transformed_data", "pca_data_400_400_100_train.npy")
read_file_test = os.path.join("..", "..", "data", "pca_transformed_data", "pca_data_400_400_100_test.npy")

def readPCA(file):
    return np.load(file)

def makeBinaryLabels(y):
    return np.array([1. if d == 2 else d for d in y])
    
def visualizePCA(file, save=True):
    pca_data = readPCA(file)
    
    x_vals = pca_data[:,0]
    y_vals = pca_data[:,1]
    z_vals = pca_data[:,2]
    
    y = makeBinaryLabels(pca_data[:, -1])
    
    fig = plt.figure(figsize=(16,9), dpi=100)
    ax = fig.add_subplot(projection='3d')
    
    ax.scatter(x_vals[y==1], y_vals[y==1], z_vals[y==1], c='#3182bd', label='1', s=3)
    ax.scatter(x_vals[y==0], y_vals[y==0], z_vals[y==0], c='#e6550d', label='0', s=3)
    
    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')
    ax.set_zlabel('PCA 3')
    
    fig.savefig('pca_visualization.png')
    print("Visualized PCA")

def kMeans(train_file, test_file, save=True):
    pca_data_train = readPCA(train_file)
    pca_data_test = readPCA(test_file)
    
    X = pca_data_train[:, 0:100]
    y = makeBinaryLabels(pca_data_train[:, -1])
    
    # Fit kMeans
    print("Fitting kMeans clustering...")
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(X)
    y_pred = kmeans.labels_

    X_test = pca_data_test[:, :100]
    y_test = makeBinaryLabels(pca_data_test[:, -1])
    test_pred = kmeans.predict(X_test)
    
    # Compute Accuracy
    permutations = [[0,1], [1,0]]
    acc_train = []
    for perm in permutations:
        y_perm = np.choose(y_pred, perm)
        acc_train.append(sum([y[i] == y_perm[i] for i in range(len(y))])/len(y))
    print("Accuracy of kMeans (train)= {:.3f}".format(max(acc_train)))

    permutations = [[0.0,1.0], [1.0,0.0]]
    acc_test = []
    for perm in permutations:
        y_perm = np.choose(test_pred, perm)
        acc_test.append(sum([y_test[i] == y_perm[i] for i in range(len(y_test))])/len(y_test))
    print("Accuracy of kMeans (test) = {:.3f}".format(max(acc_test)))
    
    # Generate Confusion Matrix - train
    perm = permutations[acc_train.index(max(acc_train))]
    y_matched = np.choose(y_pred, np.array([0.0,1.0], dtype=np.float64))
    cm_train = confusion_matrix(y, y_matched)
    
    cm_plot_train = sns.heatmap(cm_train, square=True, annot=True, fmt='d', cbar=True, cmap="YlGnBu", xticklabels=[0,1], yticklabels=[0,1])  
    if save:
        fig1 = cm_plot_train.get_figure()
        fig1.savefig("kmeans_confusionmatrix_train.png")
    fig1.clf()
    
    # Generate Confusion Matrix - test
    perm = permutations[acc_test.index(max(acc_test))]
    y_matched = np.choose(test_pred, np.array([0.0,1.0], dtype=np.float64))
    cm_test = confusion_matrix(y_test, y_matched)
    
    cm_plot_test = sns.heatmap(cm_test, square=True, annot=True, fmt='d', cbar=True, cmap="YlGnBu", xticklabels=[0,1], yticklabels=[0,1])  
    if save:
        fig2 = cm_plot_test.get_figure()
        fig2.savefig("kmeans_confusionmatrix_test.png")
    
        
if __name__ == "__main__":
    kMeans(read_file_train, read_file_test)
    visualizePCA(read_file_test)

