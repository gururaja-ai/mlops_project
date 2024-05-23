import bentoml
import hydra
import joblib
from hydra.utils import to_absolute_path as abspath
from omegaconf import DictConfig
from bentoml import BentoService, api, env, artifacts
from bentoml.adapters import DataframeInput
from bentoml.frameworks.sklearn import SklearnModelArtifact

def load_model(model_path: str):
    return joblib.load(model_path)


@hydra.main(config_path="../../config", config_name="main")
def save_to_bentoml(config: DictConfig):
    model = load_model(abspath(config.model.path))
    # bentoml.picklable_model.save(config.model.name, model)
    bentoml.xgboost.save_model(config.model.name, model)

if __name__ == "__main__":
    save_to_bentoml()
