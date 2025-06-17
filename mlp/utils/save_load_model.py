import pickle

def save_model(model, path):
    with open(path, 'wb') as f:
        pickle.dump(model, f)

def load_model(path):
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model