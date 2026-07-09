import kagglehub

# Download latest version
path = kagglehub.dataset_download("salemhafyz2/afnd-clean-parquet")

print("Path to dataset files:", path)