"""
CVAE Integration Example with Hardness-Aware Training

Evaluation on imbalanced medical datasets

12/06/2025 -> it is being executed in the screen: "CVAE"

We will execute in node 5 or 6 -> "MED-CVAE"


This module demonstrates how to integrate the hardness module with a CVAE
for tabular medical data synthesis with imbalanced datasets.
"""
import os
import itertools
import torch
from synthetic_data_evaluator import SyntheticDataEvaluator
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import StandardScaler 

import pandas as pd
from imblearn.datasets import fetch_datasets
from torch.utils.data import DataLoader, TensorDataset
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.naive_bayes import GaussianNB
# from sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split
from classifier_eval import evaluate_classification_model
from typing import Tuple, Optional
from hardness_module_improved_ import HardnessCalculator, CVAEHardnessIntegrator

# device related -> GPU or CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

class TabularCVAE(nn.Module):
    """
    Conditional Variational Autoencoder for tabular data.
    """
    
    def __init__(self, input_dim: int, latent_dim: int, condition_dim: int, 
                 hidden_dims: list = [128, 64]):
        super(TabularCVAE, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim + condition_dim
        
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent space
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)
        
        # Decoder
        decoder_layers = []
        prev_dim = latent_dim + condition_dim
        
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input with condition to latent parameters."""
        x_c = torch.cat([x, c], dim=1)
        h = self.encoder(x_c)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Decode latent representation with condition."""
        z_c = torch.cat([z, c], dim=1)
        return self.decoder(z_c)
    
    def forward(self, x: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through CVAE."""
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, c)
        return x_recon, mu, logvar


class HardnessAwareCVAETrainer:
    """
    Trainer for hardness-aware CVAE with integrated hardness scoring.
    """
    
    def __init__(self, model: TabularCVAE, hardness_calculator: HardnessCalculator,
                 hardness_integrator: CVAEHardnessIntegrator, device: str = DEVICE):
        self.model = model.to(device)
        self.hardness_calculator = hardness_calculator
        self.hardness_integrator = hardness_integrator
        self.device = device
        
        self.optimizer = optim.Adam(model.parameters(), lr=1e-3)
        self.hardness_scores = None
        
    def calculate_hardness_scores(self, X: np.ndarray, y: np.ndarray, 
                                  hardness_metrics: list, metric_index) -> np.ndarray:
        """Calculate and store hardness scores for the dataset."""
        if hardness_metrics[metric_index] is not None:
            hardness_df = self.hardness_calculator.calculate_hardness_scores(X, y, hardness_metrics)
            
            # Use the first metric as primary hardness score (can be modified)
            primary_metric = hardness_metrics[metric_index] # later we want to loop over the hardness metrics to get results with each hardness metric
            print(f"Using hardness metric in HardnessAwareCVATrainer: {primary_metric}")
            self.hardness_scores = hardness_df[primary_metric].values
            # We can return the whole dataset and later for the weight we 
            return self.hardness_scores
        else:
            self.hardness_scores = None
    
    def cvae_loss(self, x_recon: torch.Tensor, x: torch.Tensor, 
                  mu: torch.Tensor, logvar: torch.Tensor, 
                  weights: Optional[torch.Tensor] = None, 
                  beta: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate CVAE loss with optional hardness weighting.
        
        Args:
            x_recon: Reconstructed input
            x: Original input
            mu: Latent mean
            logvar: Latent log variance
            weights: Sample weights based on hardness
            beta: Beta parameter for beta-VAE
        """
        # Reconstruction loss (MSE)
        recon_loss = torch.sum((x - x_recon) ** 2, dim=1)
        
        # Apply hardness weights if provided
        if weights is not None:
            recon_loss = recon_loss * weights
        
        recon_loss = torch.mean(recon_loss)
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        kl_loss = torch.mean(kl_loss)
        
        # Total loss
        total_loss = recon_loss + beta * kl_loss
        
        return total_loss, recon_loss, kl_loss
    
    def train_epoch(self, dataloader: DataLoader, epoch: int, total_epochs: int, 
                    beta: float = 1.0) -> dict:
        """Train for one epoch with hardness-aware weighting."""
        self.model.train()
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        
        for batch_idx, (data, conditions, indices) in enumerate(dataloader):
            data = data.to(self.device)
            conditions = conditions.to(self.device)
            
            # Get hardness weights for this batch
            if self.hardness_scores is not None:
                batch_hardness = self.hardness_scores[indices.numpy()]
                weights = self.hardness_integrator.get_sample_weights(
                    batch_hardness, epoch, total_epochs
                )
                weights = torch.tensor(weights, dtype=torch.float32, device = self.device).to(self.device)
            else:
                weights = None
            
            # Forward pass
            x_recon, mu, logvar = self.model(data, conditions)
            
            # Calculate loss
            loss, recon_loss, kl_loss = self.cvae_loss(
                x_recon, data, mu, logvar, weights, beta
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
        
        n_batches = len(dataloader)
        return {
            'total_loss': total_loss / n_batches,
            'recon_loss': total_recon_loss / n_batches,
            'kl_loss': total_kl_loss / n_batches
        }
    
    def generate_samples(self, conditions: torch.Tensor, n_samples: int) -> torch.Tensor:
        """Generate synthetic samples given conditions."""
        self.model.eval()
        
        with torch.no_grad():
            # Sample from latent space
            z = torch.randn(n_samples, self.model.latent_dim).to(self.device)
            
            # Repeat conditions for all samples
            if conditions.dim() == 1:
                conditions = conditions.unsqueeze(0)
            conditions = conditions.repeat(n_samples, 1).to(self.device)
            
            # Generate samples
            synthetic_data = self.model.decode(z, conditions)
        
        return synthetic_data


# This is just for testing the code and debugging -> later we will use imblearn datasets and medical datasets:
# def create_sample_dataset(n_samples: int = 1000, n_features: int = 10, 
#                          imbalance_ratio: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
#     """Create a sample imbalanced dataset for testing."""
#     np.random.seed(42)
    
#     # Generate features
#     X = np.random.randn(n_samples, n_features)
    
#     # Create imbalanced binary labels
#     n_minority = int(n_samples * imbalance_ratio)
#     y = np.concatenate([
#         np.ones(n_minority),  # Minority class
#         np.zeros(n_samples - n_minority)  # Majority class
#     ])
    
#     # Shuffle
#     indices = np.random.permutation(n_samples)
#     X = X[indices]
#     y = y[indices]
    
#     return X, y

def prepare_dataloader(X: np.ndarray, y: np.ndarray, batch_size: int = 32) -> DataLoader:
    """Prepare DataLoader with indices for hardness tracking."""
    # Convert to tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    print(f"shape of X_tensor: {X_tensor.shape}")
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # Condition
    print(f"shape of y_tensor: {y_tensor.shape}")
    indices_tensor = torch.arange(len(X))
    
    # Create dataset and dataloader
    dataset = TensorDataset(X_tensor, y_tensor, indices_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader


def load_data(path_processed, test_size = 0.2, random_state=None):
    # Read the data from a CSV file
    df = pd.read_csv(path_processed, sep=',')
    
    # Separate features and labels
    df_base = df.iloc[:, :-1]  # Features
    df_labels = df.iloc[:, -1].values  # Labels
    print(f"Count of the labels: {pd.Series(df_labels).value_counts()}")
    print(f"Shape of the features: {df_base.shape}")
    # Split data into train and test sets using stratified random sampling
    X_train, X_test,y_train, y_test = train_test_split(
        df_base.values,
        df_labels,
        test_size=test_size,
        stratify=df_labels,
        random_state=random_state
    )
    
    # Fit the StandardScaler on training data and transform both train and test sets
    standardizer = StandardScaler()
    X_train = standardizer.fit_transform(X_train)  # Fit and transform training data
    X_test = standardizer.transform(X_test)       # Transform test data only
    
    # Identify majority and minority classes in the training set
    label_counts = pd.Series(y_train).value_counts()
    #print(label_counts)
    minority_label = label_counts.idxmin()
    majority_label = label_counts.idxmax()

    # Calculate imbalance ratio
    minority_count = label_counts[minority_label]
    majority_count = label_counts[majority_label]
    imbalance_ratio = minority_count / majority_count
    
    # Separate the training data into majority and minority classes
    minority_data = X_train[y_train == minority_label]
    majority_data = X_train[y_train == majority_label]
    
    # Print imbalance ratio
    #print(f"Imbalance Ratio (IR): {imbalance_ratio:.2f} (Majority: {majority_count}, Minority: {minority_count})")
    #print(f"type of minority data: {type(minority_data)}")
    # Return train/test sets, scaler, and separated majority/minority data
    return X_train, y_train, X_test, y_test, standardizer, majority_data, minority_data, imbalance_ratio

import random
N_EPOCHS = 150  # Total number of epochs for training
CURRICULUM_EPOCHS = (N_EPOCHS*0.3, N_EPOCHS*0.3, N_EPOCHS*0.4)  # Epochs for each hardness strategy
MASTER_SEED = 42  # Master seed for reproducibility
random.seed(MASTER_SEED)  # Random state for reproducibility
random_seeds = random.sample(range(1, 10**6), 1)  # Random seeds for different runs FIX IT TO 5 RANDOM SEEDS

def main():
    # PART 1
    # datasets = ['NewThyroid2', 'Pima', 'Thoracic', 'Vertebral']

    # PART 2
    # datasets = ['Hypothyroid', 'ILPD', 'KidneyDisease', 'NewThyroid1']

    # PART 3
    # datasets = ['BCWDD',  'Haberman', 'HeartCleveland', 'Hepatitis']
    datasets = [    'BCWDD',  'Haberman', 'HeartCleveland', 'Hepatitis', 'Hypothyroid', 'ILPD', 'KidneyDisease', 'NewThyroid1', 'NewThyroid2', 'Pima', 'Thoracic', 'Vertebral']

    weighting_strategies = ['curriculum']#, 'static', 'self_paced']
    hardness_metrics= ['feature_kDN']
    # hardness_metrics = [None, 'relative_entropy','pca_contribution', 'feature_kDN', 'feature_DS', 'feature_DCP', 'feature_TD_P',
                #    'feature_TD_U', 'feature_CL', 'feature_CLD', 'feature_MV', 'feature_CB', 'feature_N1', 'feature_N2', 'feature_LSC', 
                #    'feature_LSR', 'feature_Harmfulness', 'feature_F1', 'feature_F2', 'feature_F3', 'feature_F4']
    seeds = list(random_seeds)  # Different seeds for reproducibility CHANGE THIS 


    # Create the csv files of RESULTS_ALL and KS_RESULTS and update them with the results of each dataset and hardness metric on the airflow
    # Store results for all datasets and hardness metrics
    RESULTS_DIR = "results_medical"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plots_dir = f"{RESULTS_DIR}/plots"
    os.makedirs(plots_dir, exist_ok=True)

    # ✅ Create empty CSV files for all results and KS results if they don’t exist
    all_results_path = os.path.join(RESULTS_DIR, 'all_classification_results_medical_1.csv')
    ks_results_path = os.path.join(RESULTS_DIR, 'all_ks_results_medical_1.csv')

    if not os.path.exists(all_results_path):
        pd.DataFrame().to_csv(all_results_path, index=False)

    if not os.path.exists(ks_results_path):
        pd.DataFrame().to_csv(ks_results_path, index=False)

    fid_save_dir = "fidelity_results_medical"
    os.makedirs(fid_save_dir, exist_ok=True)

    fidelity_views = {'statistical':{}, 'hardness':{}, 'complexity':{}, 'clustering':{}, 'topological':{}}

    # for key in fidelity_views.keys():

    #     if not os.path.exists(f"ks_results_path"):
    #         pd.DataFrame().to_csv(ks_results_path, index=False)
    # Generate valid combinations
    combinations = []
    for dataset_name, hardness_metric, seed in itertools.product(datasets, hardness_metrics, seeds):
        if hardness_metric is None:
            strategies = ['static']
        else:
            strategies = ['curriculum', 'static', 'self_paced']
        for strategy in strategies:
            combinations.append((dataset_name, hardness_metric, seed, strategy))

    # Loop through valid combinations
    # fidelity_views = {'statistical':{}, 'hardness':{}, 'complexity':{}, 'clustering':{}, 'topological':{}}

    for dataset_name, hardness_metric, seed, weighting_strategy in combinations: #itertools.product(datasets, hardness_metrics, seeds, weighting_strategies):
        print(f"\n=== Dataset: {dataset_name} | hardness metric: {hardness_metric} | Weighting strategy: {weighting_strategy} | Seed: {seed} ===")
        print("="*50)
        print("\n=== Step 1: Loading dataset ===")
        # Path of the plots directory for the synthetic data evaluator: plots are specific to each combination and dataset so needs to make specific directory for each combination plots:
        plot_dir = f"{plots_dir}/plots_{hardness_metric}_{weighting_strategy}_{seed}"
        os.makedirs(plot_dir, exist_ok=True)
        # # Load dataset
        # data = fetch_datasets()[dataset_name]
        # X, y = data.data, data.target
        # # Label encode y to be in 0 or 1 instead of -1, 1
        # y = np.where(y == -1, 0, 1)

        # Train-test split
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
        DATA_PATH = f'inputs/{dataset_name}_processed.csv'
        print(f"DATASET: {dataset_name}")
        X_train, y_train, X_test, y_test, _, _, _, _ = load_data(DATA_PATH, random_state=seed)

        # hardness_calc = HardnessCalculator(random_state=seed)
        # print("2. Calculating hardness scores...")


    
        # # Initialize CVAE model
        # print("3. Initializing CVAE model...")
        # input_dim = X_train.shape[1]
        # latent_dim = 5
        # condition_dim = 1  # Binary condition
        
        # model = TabularCVAE(
        #     input_dim=input_dim,
        #     latent_dim=latent_dim,
        #     condition_dim=condition_dim,
        #     hidden_dims=[128, 64, 32]
        # )
        # # print(f"Model parameters: {sum(p.numel() for p in model.parameters())}\\n")
        
        # # Initialize hardness integrator
        # print("4. Setting up hardness-aware training...")
        # hardness_integrator = CVAEHardnessIntegrator(
        #     hardness_strategy= weighting_strategy # 'curriculum',    #'curriculum',
        #     #curriculum_epochs= CURRICULUM_EPOCHS,  # (N_EPOCHS*0.3, N_EPOCHS*0.3, N_EPOCHS*0.4)
        # )
        
        # # Initialize trainer
        # trainer = HardnessAwareCVAETrainer(
        #     model=model,
        #     hardness_calculator=hardness_calc,
        #     hardness_integrator=hardness_integrator,
        #     device=DEVICE
        # )
        
        # # Calculate hardness scores for training
        # trainer.calculate_hardness_scores(X_train, y_train, [hardness_metric], 0)
        
        # # Prepare data
        # dataloader = prepare_dataloader(X_train, y_train, batch_size=32)
        
        # # Training loop
        # print("5. Training CVAE with hardness awareness...")
        # n_epochs = N_EPOCHS
        
        # for epoch in range(n_epochs):
        #     metrics = trainer.train_epoch(dataloader, epoch, n_epochs, beta=1.0)
            
        #     if epoch % 10 == 0:
        #         print(f"Epoch {epoch:2d}: Loss={metrics['total_loss']:.4f}, "
        #             f"Recon={metrics['recon_loss']:.4f}, KL={metrics['kl_loss']:.4f}")
        
        # print("\\n6. Generating synthetic samples...")

        # # Define the number of samples to generate: difference in the count of the classes
        # # first define the majority class label and count and the same for the minority class
        # majority_class_label = np.argmax(np.bincount(y_train))
        # print(f"majority class label : {majority_class_label}")
        # minority_class_label = np.argmin(np.bincount(y_train))
        # print(f"minority class label: {minority_class_label}")

        # majority_class_count = np.bincount(y_train)[majority_class_label]
        # print(f"Majority Count: {majority_class_count}")
        # minority_class_count = np.bincount(y_train)[minority_class_label]
        # print(f"Minority Count: {minority_class_count}")
        # N_SAMPLES = majority_class_count - minority_class_count # Number of minority samples needed
        # print(f"N_SAMPLES:{N_SAMPLES}")
        # # Generate samples for minority class (condition = 1)
        # minority_condition = torch.tensor([minority_class_label])

        # synthetic_samples = trainer.generate_samples(minority_condition, n_samples = N_SAMPLES)
        # X_min_real = X_train[y_train == minority_class_label]

        # # get synthetic samples with their labels and concatenate them with the original data
        # synthetic_samp = synthetic_samples.cpu().numpy()
        # synthetic_labels = np.full((synthetic_samp.shape[0], 1), minority_class_label)
        
        # # Fidelity evaluation:
        # evaluator = SyntheticDataEvaluator(random_state=seed)
        # fid_results = evaluator.evaluate_all(
        #     X_train, y_train,
        #     synthetic_samp, synthetic_labels.reshape(-1),
        #     save_path= plot_dir, dataset_name= dataset_name
        # )
        # records = []
        # for key in fidelity_views.keys():
        #     csv_path = os.path.join(fid_save_dir, f"fidelity_results_{key}.csv")

        #     record = {            
        #         'dataset': dataset_name,
        #         'hardness_metric': hardness_metric,
        #         'seed': seed,
        #         'weighting_strategy': weighting_strategy}
        #     record.update(fid_results.get(key,{}))
        #     records.append(record)
        #     # fidelity_views[key].append(record)
        #     # Save to CSV
        #     df_new = pd.DataFrame([record])
        #     # If file exists, append without duplicating header
        #     if os.path.exists(csv_path):
        #         df_new.to_csv(csv_path, mode='a', header=False, index=False)
        #     else:
        #         df_new.to_csv(csv_path, index=False)



        # # Evaluate similarity of synthetic samples to real minority samples
        # from sdv_metrics import evaluate_synthetic_data
        # print("="*50)
        # print("="*50)
        # ks_result = evaluate_synthetic_data(X_min_real, y_train[y_train == minority_class_label], 
        #                     synthetic_samples.cpu().numpy(), 
        #                         np.full((synthetic_samples.shape[0], 1), minority_class_label), 
        #                         verbose=True)  
        # ks_record = {
        #     'dataset': dataset_name,
        #     'hardness_metric': hardness_metric,
        #     'seed': seed,
        #     'weighting_strategy': weighting_strategy,
        #     'ks_mean': ks_result['ks_mean'],
        #     'ks_std': ks_result['ks_std']
        # }
        # # append the ks_record to CSV
        # pd.DataFrame([ks_record]).to_csv(ks_results_path, mode='a', header=not os.path.exists(ks_results_path) or os.stat(ks_results_path).st_size == 0 , index=False) # 

        # print("="*50)
        # print("="*50)
        # synthetic_samp = np.concatenate([synthetic_samp, synthetic_labels], axis=1)
        # # Convert to DataFrame for better visualization
        # synthetic_samp = pd.DataFrame(synthetic_samp, columns=[f'feature_{i}' for i in range(X_train.shape[1])] + ['label'])
        # # concatenate the synthetic samples with the original data
        # original_data = pd.DataFrame(X_train, columns=[f'feature_{i}' for i in range(X_train.shape[1])])
        # original_data['label'] = y_train
        # combined_data = pd.concat([original_data, synthetic_samp], ignore_index=True)

        # # print("\\n=== Training completed successfully! ===")

        # # Now that we have the synthetic samples and the training data balanced we create a function to classify the data and evaluate the model with classification metrics
        # ## Use a bunch of classification algorithms to classify the data and evaluate the model with classification metrics
        # results_dir = f"{RESULTS_DIR}/{dataset_name}_{hardness_metric}_seed{seed}_weighting_{weighting_strategy}"
        # os.makedirs(results_dir, exist_ok=True)
        # X_aug = np.array(combined_data.iloc[:,:-1].copy())
        # print(f"shape of X_aug : {X_aug.shape}")
        # y_aug = np.array(combined_data.iloc[:,-1].copy())
        # print(f"shape of y_aug: {y_aug.shape}")
        # result = evaluate_classification_model(X_aug, y_aug, X_test, y_test, k_folds=3, hardness_metric=hardness_metric,  random_state=seed)
        
        # for r in result:
        #     r.update({
        #         'dataset': dataset_name,
        #         'hardness_metric': hardness_metric,
        #         'seed': seed,
        #         'weighting_strategy': weighting_strategy
        #     })
        
        # # Append classification results to CSV
        # # RESULTS_ALL.append(result)
        # # if the csv file of the results is empty createe the header:
        # result_cols = list(result[0].keys())
        # pd.DataFrame(result, columns=result_cols).to_csv(all_results_path, mode='a', header=not os.path.exists(all_results_path) , index=False) # or os.stat(all_results_path).st_size == 0

        
        # # Save individual results as CSV
        # experiment_dir = f"{results_dir}/{dataset_name}_{hardness_metric}_seed{seed}"
        # os.makedirs(experiment_dir, exist_ok=True)
        # pd.DataFrame(result).to_csv(os.path.join(experiment_dir, 'classification_results.csv'), index=False)



    print(f"\n✅ All runs complete. Aggregated results are saved in:\n- {all_results_path}\n- {ks_results_path}")


if __name__ == "__main__":
    main()

