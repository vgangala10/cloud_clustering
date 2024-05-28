train = {
    'model_directory': '/storage/climate-memmap/models/VAE/orig_model',
    'batch_size': 128,
    'number_of_workers': 4,
    'training_files': 90,
    'embedding_size': 50,
    'learning_rate': 1e-5,
    'epochs': 50
}

embedding = {
    'embedding_size': 50,
    'number_of_files': 10,
    'path': '/storage/climate-memmap/models/ResNet34/embedding_50_transform_files_90_land',
    'model_directory': '/storage/climate-memmap/models/ResNet34/embedding_50_transform_files_90_land/best_model-epoch=38-val_loss=0.00.ckpt',
    'model_final_path': '/storage/climate-memmap/models/ResNet34/embedding_50_transform_files_90_land/lightning_model_50_transform.pt',
    'embeddings_memmap_path': '/storage/climate-memmap/models/ResNet34/embedding_50_transform_files_90_land/test_embeddings_50_coords.memmap'
}

clustering = {
    'kmeans_path': '/storage/climate-memmap/models/ResNet34/embedding_100_transform_files_60/kmeans',
    'gaussian_path': '/storage/climate-memmap/models/ResNet34/embedding_75_transform_files_60/gaussian_mixture'
}