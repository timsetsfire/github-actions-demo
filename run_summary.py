import pandas as pd
import wandb
api = wandb.Api()
entity, project, sweep_id = "tim-w", "pytorch-sweeps-demo", "zc4rne6j"
## either pull runs for an entire project
runs = api.runs(f"{entity}/{project}")
## or pull runs for a particular sweep
sweep = api.sweep(f"{entity}/{project}/{sweep_id}")
temp_data = []
for r in sweep.runs:
 temp_dict = dict(**dict(r.summary), **r.config)
 temp_dict["run_id"] = r.id
 temp_dict["run_name"] = r.name
 temp_data.append( temp_dict)
# df = pd.DataFrame(temp_data)
# df.set_index("run_id", inplace = True)
# best_run_id = sweep.best_run().id
# print(df.loc[best_run_id])
for run in temp_data:
    print(run)