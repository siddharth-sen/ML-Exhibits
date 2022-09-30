from numpy.core.numeric import full
from sklearn.decomposition import PCA
import numpy as np
from load_data import load_data
import os

save_dir = os.path.join("..", "..", "data", "pca_transformed_data")

def pca_transformation(data,n_comp):
    data_no_labels = data[:,:-1]
    pca = PCA(n_comp)
    transformed_data = pca.fit_transform(data_no_labels)
    print("Preserved variance = {}".format(np.sum(pca.explained_variance_ratio_)))
    return np.hstack((transformed_data,data[:,-1].reshape((len(data),1))))

def pca_pipeline(resize=(400,400), n_comp=100, save=True):
    train_data = load_data(resize, set="train", testing=False)
    test_data = load_data(resize, set='test', testing=False)
    print("PCA with n_comp = {}".format(n_comp))

    train_n = train_data.shape[0]
    test_n = test_data.shape[0]
    # concatonate together ot transform together
    all_data = np.concatenate((train_data, test_data), axis=0)
    pca_trans = pca_transformation(all_data, n_comp)
    
    #get train and test samples out
    train_pca_trans = pca_trans[:train_n, :]
    test_pca_trans = pca_trans[train_n:, :]

    if save:
        train_filename = "pca_data_{}_{}_{}_{}.npy".format(resize[0], resize[1], n_comp, "train")
        full_save_path = os.path.join(save_dir, train_filename)
        with open(full_save_path, 'wb') as f:
            np.save(f, train_pca_trans)
        print("Train data saved to: {}".format(full_save_path))


        test_filename = "pca_data_{}_{}_{}_{}.npy".format(resize[0], resize[1], n_comp, "test")
        full_save_path = os.path.join(save_dir, test_filename)
        with open(full_save_path, 'wb') as f:
            np.save(f, test_pca_trans)
        print("Test data saved to: {}".format(full_save_path))
    return train_pca_trans,test_pca_trans

if __name__ == "__main__":
    pca_pipeline(resize=(400,400))