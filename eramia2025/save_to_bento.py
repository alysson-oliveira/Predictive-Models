import bentoml
import pickle

model = pickle.leados(open('data/models/model.pickle'))

saved_model = bentoml.sklearn.save_model(
    'bbb-model',
    model,
    signatures={
        "predict": {"natchable": True}
        "predict_proba": {"batchable": True}
    }
)

print(saved_model)