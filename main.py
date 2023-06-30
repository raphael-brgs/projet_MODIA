import argparse
import pandas as pd
import torch
from torch.utils.data import Dataset
from model import NCF

class RatingsDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index()
        self.user_list = self.df.user_id.unique()
        self.recipe_list = self.df.recipe_id.unique()
        self.user2id = {w: i for i, w in enumerate(self.user_list)}
        self.recipe2id = {w: i for i, w in enumerate(self.recipe_list)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        user = self.user2id[self.df.loc[index, 'user_id']]
        recipe = self.recipe2id[self.df.loc[index, 'recipe_id']]
        user_tensor = torch.tensor(user, dtype=torch.long)
        recipe_tensor = torch.tensor(recipe, dtype=torch.long)
        rating_tensor = torch.tensor(self.df.loc[index, 'rating'], dtype=torch.float)
        return user_tensor, recipe_tensor, rating_tensor

def load_model_weights(model, weights_path):
    model.load_state_dict(torch.load(weights_path,map_location=torch.device('cpu')))
    model.eval()

def predict_ratings(model, dataset):
    ratings = []
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
    for user, item, _ in dataloader:
        rating = model(user, item)
        ratings.append(rating.item())
    return ratings

def main(weights_path, test_data_path):
    # Load the test data from the CSV file
    test_data = pd.read_csv(test_data_path)

    # Create a RatingsDataset from the test data
    test_dataset = RatingsDataset(test_data)

    # Create an instance of the NCF model
    n_users = 25076 # n_users in training set
    n_recipes = 160901 # n_recipes in training set
    ncf_model = NCF(n_users=n_users, n_items=n_recipes)

    # Load the pre-trained model weights
    load_model_weights(ncf_model, weights_path)

    # Predict ratings for the test interactions
    ratings = predict_ratings(ncf_model, test_dataset)

    # Display the predictions
    for i, rating in enumerate(ratings):
        print(f"Prediction {i+1}: {rating}")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("weights_path", help="Path to the model weights file (.pth)")
    parser.add_argument("test_data_path", help="Path to the test data CSV file")
    args = parser.parse_args()

    # Call the main function
    main(args.weights_path, args.test_data_path)
