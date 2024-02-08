train = {
    'model_directory': '/storage/climate-memmap/models/ResNet34/embedding_100_transform_files_60',
    'batch_size': 32,
    'number_of_workers': 4,
    'training_files': 60,
    'embedding_size': 100,
    'learning_rate': 0.001
}

embedding = {
    'embedding_size': 100,
    'number_of_files': 5,
    'path': '/storage/climate-memmap/models/ResNet34/embedding_50_transform_files_60/',
    'model_directory': '/storage/climate-memmap/models/ResNet34/embedding_50_transform_files_60/best_model-epoch=46-val_loss=0.00.ckpt',
    'model_final_path': '/storage/climate-memmap/models/ResNet34/embedding_100_transform_files_60/lightning_model_100_transform.pt',
    'embeddings_memmap_path': '/storage/climate-memmap/models/ResNet34/embedding_50_transform_files_60/test_embeddings_50_coords.memmap'
}

clustering = {
    'kmeans_path': '/storage/climate-memmap/models/ResNet34/embedding_75_transform_files_60/kmeans',
    'gaussian_path': '/storage/climate-memmap/models/ResNet34/embedding_75_transform_files_60/gaussian_mixture'
}