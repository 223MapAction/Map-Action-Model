import os
import pandas as pd
import numpy as np
import requests
import torch
from glob import glob
from dotenv import load_dotenv
from dagshub.data_engine import datasources
from typing import Annotated, Optional, Tuple
from ..data_preprocess.data_transform import get_transform
from zenml import step

load_dotenv()

DAGSHUB_REPO_OWNER = os.environ.get("DAGSHUB_REPO_OWNER")
DAGSHUB_REPO = os.environ.get("DAGSHUB_REPO")

DAGSHUB_FULL_REPO = f"{DAGSHUB_REPO_OWNER}/{DAGSHUB_REPO}"

@step
def download_and_organize_data() -> Tuple[
    Annotated[str, "train_dir"],
    Annotated[str, "valid_dir"],
    Annotated[str, "test_dir"],
    Annotated[torch.tensor, "transform"],
    Annotated[int, "batch_size"],
    Annotated[int, "NUM_WORKERS"]
]:
    # Load data from a CSV file
    Print("step1")
    ds = datasources.get(DAGSHUB_FULL_REPO, os.environ.get("DATASOURCE_NAME"))
    ds = ds.all().dataframe
    print(ds)
    df = pd.read_csv("project-8.csv=", usecols=["choice", "image"])
    df_img = df["image"]
    ds_img = ds["path"]

    # Create directories for train, valid, and test
    data_dir = "data"
    train_dir = "train"
    valid_dir = "valid"
    test_dir = "test"

    # Randomly permute indices
    data_volumes = rand_shuffle = np.random.permutation(df_img.shape[0])
    print(data_volumes)

    print(ds_img.shape[0])

    # Download images and organize them into directories
    for n in range(df_img.shape[0]):
        for m in range(ds_img.shape[0]):
            if df_img[n].split('/')[-1] == ds_img[m]:
                save_dir = os.path.join(data_dir, df['choice'][n])
                try:
                    os.makedirs(save_dir)
                except FileExistsError:
                    pass  # Directory already exists
                response = requests.get(df_img[n], stream=True)
                with open(f"{save_dir}/images{n+1}.jpg", 'wb') as file:
                    for chunk in response.iter_content(chunk_size=128):
                        file.write(chunk)
                print(f"File downloaded successfully in {n+1}")

    # Gather file paths
    types = ('*.jpg', '*.jpeg', '*.png')
    files = []
    for ext in types:
        f = glob(os.path.join(data_dir, '*/'+ext))
        files += f
    print(files)

    # Organize files into train, valid, and test directories
    if len(files) > 0:
        for i, split in enumerate(['valid', 'test', 'train']):
            for n in data_volumes[i * 5: (i + 1) * 5]:
                folder = files[n].split('/')[2]
                name = files[n].split('/')[-1]
                try:
                    os.makedirs(os.path.join(data_dir, split, folder))
                except FileExistsError:
                    pass  # Directory already exists
                os.rename(files[n], os.path.join(data_dir, split, folder, name))
                
    print(f"{data_dir}/{train_dir}")
    train_dir = f"{data_dir}/{train_dir}"
    valid_dir = f"{data_dir}/{valid_dir}"
    test_dir = f"{data_dir}/{test_dir}"
    batch_size = 20
    NUM_WORKERS = 2
    transform = get_transform(train=True)

    return  train_dir, valid_dir, test_dir, transform, batch_size, NUM_WORKERS
    


