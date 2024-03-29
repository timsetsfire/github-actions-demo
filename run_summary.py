import wandb
import pandas as pd
import logging
from convnet import *
import yaml 

with open("model_conf.yaml", "r") as f:
  model_config = yaml.load(f, Loader = yaml.FullLoader)

logging.basicConfig(
        format="{} - %(levelname)s - %(asctime)s - %(message)s".format(__name__),
)
logger = logging.getLogger("git-action-example")
logger.setLevel("INFO")

# api = wandb.Api()
# entity, project_name = "tim-w", "MNIST-Training"
# device = torch.device("cpu")
# ## either pull runs for an entire project
# runs = [r for r in api.runs(f"{entity}/{project_name}") if r.job_type == "training" or "sweep" in r.name]
# ## or pull runs for a particular sweep
# temp_data = []
# for r in runs:
#  temp_dict = dict(**dict(r.summary), **r.config)
#  temp_dict["run_id"] = r.id
#  temp_dict["run_name"] = r.name
#  temp_data.append( temp_dict)
# df = pd.DataFrame(temp_data).sort_values(by = "Validation Metrics/loss")
# # df.set_index("run_id", inplace = True)
# best_run_id = df["run_id"].iloc[0]
# print(df.set_index("run_id").loc[best_run_id])
# best_run = [r for r in runs if r.id == best_run_id].pop()
project_name = model_config["project_name"]
entity = model_config["entity"]
device = torch.device("cpu")
with wandb.init(project = project_name, name = "best_model_eval", entity = entity, job_type = "evaluation", config = model_config) as run:
  
  config = wandb.config 
  
  ## load model
  model_artifact = run.use_artifact(config.usage_string, type = "model")
  model_artifact_directory = model_artifact.download()
  model = ConvNet(wandb.config.kernels, wandb.config.classes)
  model.load_state_dict(torch.load(model_artifact.file()))
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
  
  ##
  dataset_artifact = run.use_artifact(f"{entity}/{project_name}/mnist-test-data:latest")
  dataset_dir = dataset_artifact.download("./data")
  test = torch.load(dataset_artifact.file())
  test_loader = make_loader(test, batch_size=config.batch_size)
  
  ## same goes for the dataset
  test_loader = make_loader(test, batch_size=config.batch_size)

  model.eval()
  # Run the model on some test examples

  with torch.no_grad():
      correct, total = 0, 0
      total_loss = 0
      all_data = []
      for images, labels in test_loader:
          images, labels = images.to(device), labels.to(device)
          outputs = model(images)
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum().item()
          loss = criterion(outputs, labels)*labels.size(0)
          total_loss += loss
          wandb_images = []
          for image in images.numpy():
            temp = wandb.Image(image)
            wandb_images.append(temp) 
          scores = pd.DataFrame( outputs.numpy().tolist(), columns = [f"p{i}" for i in range(outputs.shape[1])]).to_dict(orient = "series")
          data = {"images":wandb_images, "predicted": predicted.numpy().tolist(), "labels": labels.numpy().tolist()}
          data = {**data, **scores}
          all_data.append(pd.DataFrame(data))
      import pandas as pd 
      df = pd.concat(all_data)
      wandb.log({"Predictions vs Actuals": wandb.Table(dataframe = df)})
      run.log({"Test Metrics/loss": total_loss / total, "Test Metrics/accuracy": correct / total})
      logger.info(f"Accuracy of the model on the {total} " +
            f"test images: {100 * correct / total}%")
          
          
