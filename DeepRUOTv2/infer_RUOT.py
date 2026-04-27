# Generate outputs needed for metrics: sde_point_*.npy, sde_weight_*.npy, velocity.h5ad, etc.
# sde_point_*.npy, sde_weight_*.npy

# velocity.h5ad

# pseudotime-related outputs (downstream)

# type: ignore

# evaluate RUOT and visulazation
import os
import sys
import argparse
import random
import math
import joblib
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import anndata
import scvelo as scv
import scanpy as sc
import umap
from tqdm import tqdm
from sklearn.decomposition import PCA
import ot
from torchdiffeq import odeint_adjoint as odeint

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

from DeepRUOT.losses import OT_loss1
from DeepRUOT.utils import (load_and_merge_config, euler_sdeint, euler_sdeint_reverse,)
from DeepRUOT.models import FNet, scoreNet2
from DeepRUOT.constants import DATA_DIR, RES_DIR
from DeepRUOT.exp import setup_exp


class SDE(torch.nn.Module):
    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self, ode_drift, g, score, input_size=(3, 32, 32), sigma=1.0):
        super().__init__()
        self.drift = ode_drift
        self.score = score
        self.input_size = input_size
        self.sigma = sigma
        self.g_net = g

    # Drift
    def f(self, t, y):
        z, lnw = y
        drift=self.drift(t, z)
        dlnw = self.g_net(t, z)
        num = z.shape[0]
        t = t.expand(num, 1)  # Keep gradient information of t and expand its shape
        return (drift+self.score.compute_gradient(t, z), dlnw)

    # Diffusion
    def g(self, t, y):
        return torch.ones_like(y)*self.sigma
    

class SDE_reverse(torch.nn.Module):
    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self, ode_drift, g, score, input_size=(3, 32, 32), sigma=1.0):
        super().__init__()
        self.drift = ode_drift
        self.score = score
        self.input_size = input_size
        self.sigma = sigma
        self.g_net = g

    # Drift
    def f(self, t, y):
        z, lnw = y
        drift=self.drift(t, z)
        dlnw = self.g_net(t, z)
        num = z.shape[0]
        t = t.expand(num, 1)  # Keep gradient information of t and expand its shape
        if self.sigma == 0:
            return (drift, dlnw)
        else:
            return (drift - self.score.compute_gradient(t, z), dlnw)

    # Diffusion
    def g(self, t, y):
        return torch.ones_like(y)*self.sigma
    

def load_data_model(config, device):
    """Load CSV data and trained FNet / score networks."""
    df = pd.read_csv(os.path.join(DATA_DIR, config['data']['file_path']))
    df = df.iloc[:, :config['data']['dim'] + 1]
    exp_dir = os.path.join(config['exp']['output_dir'], config['exp']['name'])
    model_config = config['model']
    f_net = FNet(
        in_out_dim=model_config['in_out_dim'],
        hidden_dim=model_config['hidden_dim'],
        n_hiddens=model_config['n_hiddens'],
        activation=model_config['activation']
    ).to(device)
    sf2m_score_model = scoreNet2(
        in_out_dim=model_config['in_out_dim'],
        hidden_dim=model_config['score_hidden_dim'],
        activation=model_config['activation']
    ).float().to(device)
    f_net.load_state_dict(torch.load(os.path.join(exp_dir, 'model_final'), map_location=device))
    sf2m_score_model.load_state_dict(torch.load(os.path.join(exp_dir, 'score_model_final'), map_location=device))
    return df, f_net, sf2m_score_model, exp_dir


def generate_velocity(df, f_net, dim, exp_dir, device='cuda'):
    """
    generate velocity data
    """
    all_times = df['samples'].values
    all_data = df[[f'x{i}' for i in range(1, dim + 1)]].values
    # Step 2: Convert to PyTorch tensors
    t_tensor = torch.tensor(all_times).unsqueeze(1).float().to(device)
    data_tensor = torch.tensor(all_data, dtype=torch.float32).to(device)
    data_tensor.requires_grad_(True)
    gradients = f_net.v_net(t_tensor, data_tensor)
    data_np = data_tensor.detach().cpu().numpy()
    data_np_2d = data_np[:, :2]
    
    gradients_np = gradients.detach().cpu().numpy()
    data_end = data_np + gradients_np

    data_end_2d = data_end[:, :2]
    gradients_np = data_end_2d - data_np_2d
    gradients_np = gradients_np / np.linalg.norm(gradients_np, axis=1, keepdims=True) * 5
    adata = anndata.AnnData(X=all_data)
    adata.layers['Ms'] = all_data 
    adata.obsm['X_PCA'] = data_np_2d
    adata.obs['time'] = all_times
    adata.layers['velocity'] = gradients.detach().cpu().numpy()

    adata.write_h5ad(os.path.join(exp_dir, 'velocity.h5ad'))

    return adata


def generate_trajectory_point(df, f_net, sf2m_score_model, dim, exp_dir, sigma, initial_num_points = None, device = 'cuda', traject_sample_number = 100, num_runs = 5, use_mass = False, output_dir = None, file_prefix = "sde"):
    """
    generate simulated data, 
        point data & weight data for dynamic metric calculation(W1, TMV), 
        trajectory data & ts & lnw for pesudotime calculation
    """
    all_times = df['samples'].values
    all_data = df[[f'x{i}' for i in range(1, dim + 1)]].values

    n_times = all_times.max() + 1
    data=torch.tensor(df[df['samples']==0].values,dtype=torch.float32).requires_grad_()
    data_t0 = data[:, 1:].to(device).requires_grad_()
    print(data_t0.shape)
    x0=data_t0.to(device)
    
    # Roll out from t0
    results = []
    for run_idx in tqdm(range(num_runs)):
        SEED = run_idx
        random.seed(SEED)
        np.random.seed(SEED)
        torch.random.manual_seed(SEED)
        if initial_num_points is not None:
            sample_indices = random.sample(range(x0.size(0)), initial_num_points)
            x0_subset = x0[sample_indices].to(device)
        else:
            x0_subset = x0.to(device)

        lnw0 = torch.log(torch.ones(x0_subset.shape[0], 1) / x0_subset.shape[0]).to(device)
        initial_state = (x0_subset, lnw0)

        for param in f_net.parameters():
            param.requires_grad = False
        for param in sf2m_score_model.parameters():
            param.requires_grad = False

        # Define SDE object
        sde = SDE(f_net.v_net, 
                f_net.g_net, 
                sf2m_score_model, 
                input_size=(2,), 
                sigma=sigma)

        ts = torch.linspace(0, n_times - 1, 100, device=device)
        sde_traj, traj_lnw = euler_sdeint(sde, initial_state, dt=0.1, ts=ts)  # sample trajectories from t0
        sde_traj, traj_lnw = sde_traj.cpu(), traj_lnw.cpu()

        sample_number = traject_sample_number  # number of trajectories to save
        sample_indices = random.sample(range(sde_traj.size(1)), sample_number)
        sampled_sde_trajec = sde_traj[:, sample_indices, :]
        sampled_sde_trajec = sampled_sde_trajec.tolist()
        sampled_sde_trajec = np.array(sampled_sde_trajec, dtype=object)
        np.save(os.path.join(exp_dir, f'sde_trajec_{run_idx}.npy'), sampled_sde_trajec)

        ts_points = torch.tensor(sorted(df.samples.unique()), dtype=torch.float32)
        sde_point, traj_lnw = euler_sdeint(sde, initial_state, dt=0.1, ts=ts_points)
        if use_mass:
            weight = torch.exp(traj_lnw)
        else:
            weight = torch.ones_like(traj_lnw)

        sde_point_np = sde_point.detach().cpu().numpy()
        sde_point_list = sde_point_np.tolist()
        sde_point_array = np.array(sde_point_list, dtype=object)
        np.save(os.path.join(exp_dir, f'sde_point_{run_idx}.npy'), sde_point_array)
        np.save(os.path.join(exp_dir, f'sde_weight_{run_idx}.npy'), weight.detach().cpu().numpy())

        out = {
            "traj": sde_traj.detach().cpu().numpy(),  # (T,N,G)
            "lnw": traj_lnw.detach().cpu().numpy(),   # (T,N,1)
            "ts": ts.detach().cpu().numpy(),          # (T,)
            "point": sde_point_array,  # (T,N,G)
            "weight": weight.detach().cpu().numpy(),   # (T,N,1)
        }
        results.append(out)

        if output_dir is not None:
            np.savez(
                os.path.join(output_dir, f"{file_prefix}_run_{run_idx}.npz"),
                traj=out["traj"],
                lnw=out["lnw"],
                ts=out["ts"],
                point=out["point"],
                weight=out["weight"]
            )

    return results


def generate_trajectories_sde_hold_start_out(df, f_net, sf2m_score_model, device, exp_dir, all_times, sigma, use_mass, num_points=None, num_runs = 1, output_dir = None, file_prefix = "sde"):
    n_times = np.array(all_times).max() + 1
    data=torch.tensor(df[df['samples']==all_times[1]].values,dtype=torch.float32).requires_grad_() # t1 data
    data_t1 = data[:, 1:].to(device).requires_grad_()
    x1=data_t1.to(device)
    results = []
    for run_idx in tqdm(range(num_runs)):
        SEED = run_idx
        random.seed(SEED)
        np.random.seed(SEED)
        torch.random.manual_seed(SEED)
        if num_points is not None:
            sample_indices = random.sample(range(x1.size(0)), num_points)
            x1_subset = x1[sample_indices].to(device)
        else:
            x1_subset = x1.to(device)

        lnw0 = torch.log(torch.ones(x1_subset.shape[0], 1) / x1_subset.shape[0] * (len(df[df['samples'] == all_times[1]]) / len(df[df['samples'] == all_times[0]]))).to(device)
        initial_state = (x1_subset, lnw0)

        for param in f_net.parameters():
            param.requires_grad = False
        for param in sf2m_score_model.parameters():
            param.requires_grad = False

        # Define SDE object
        sde = SDE(f_net.v_net, 
                f_net.g_net, 
                sf2m_score_model, 
                input_size=(2,), 
                sigma=sigma)
        sde_reverse = SDE_reverse(f_net.v_net, 
                f_net.g_net, 
                sf2m_score_model, 
                input_size=(2,), 
                sigma=sigma)
        
        ts_reverse = torch.linspace(all_times[1], all_times[0], 10, device=device)
        sde_traj_reverse, traj_lnw_reverse = euler_sdeint_reverse(sde_reverse, initial_state, dt=0.1, ts=ts_reverse)
        sde_traj_reverse, traj_lnw_reverse = sde_traj_reverse.cpu(), traj_lnw_reverse.cpu()

        ts_forward = torch.linspace(all_times[1], n_times - 1, 100, device=device)
        sde_traj_forward, traj_lnw_forward = euler_sdeint(sde, initial_state, dt=0.1, ts=ts_forward)
        sde_traj_forward, traj_lnw_forward = sde_traj_forward.cpu(), traj_lnw_forward.cpu()

        sde_traj = torch.cat([sde_traj_reverse.flip(0), sde_traj_forward[1:]], dim=0)
        traj_lnw = torch.cat([traj_lnw_reverse.flip(0), traj_lnw_forward[1:]], dim=0)

        sample_number = 100  # For example, sample 10
        sample_indices = random.sample(range(sde_traj.size(1)), sample_number)
        sampled_sde_trajec = sde_traj[:, sample_indices, :]
        sampled_sde_trajec = sampled_sde_trajec.tolist()
        sampled_sde_trajec = np.array(sampled_sde_trajec, dtype=object)
        np.save(os.path.join(exp_dir, f'sde_trajec_{run_idx}.npy'), sampled_sde_trajec)

        ts_points_reverse = torch.tensor([all_times[1], all_times[0]], dtype=torch.float32)
        sde_point_0, traj_lnw_0 = euler_sdeint_reverse(sde_reverse, initial_state, dt=0.1, ts=ts_points_reverse)
        if use_mass:
            weight_0 = torch.exp(traj_lnw_0)
        else:
            weight_0 = torch.ones_like(traj_lnw_0)

        sde_point_np_0 = sde_point_0[-1].unsqueeze(0).detach().cpu().numpy()
        weight_0 = weight_0[-1].unsqueeze(0).detach().cpu().numpy()

        ts_points_forward = torch.tensor([all_times[t] for t in range(1, len(all_times))], dtype=torch.float32)
        print(ts_points_forward)
        sde_point_forward, traj_lnw_forward = euler_sdeint(sde, initial_state, dt=0.1, ts=ts_points_forward)
        if use_mass:
            weight_forward = torch.exp(traj_lnw_forward)
        else:
            weight_forward = torch.ones_like(traj_lnw_forward)

        sde_point_np_forward = sde_point_forward.detach().cpu().numpy()
        weight_forward = weight_forward.detach().cpu().numpy()
        sde_point_np = np.concatenate([sde_point_np_0, sde_point_np_forward], axis=0)
        print(sde_point_np.shape)
        weight = np.concatenate([weight_0, weight_forward], axis=0)
        sde_point_list = sde_point_np.tolist()
        sde_point_array = np.array(sde_point_list, dtype=object)
        np.save(os.path.join(exp_dir, f'sde_point_{run_idx}.npy'), sde_point_array)
        np.save(os.path.join(exp_dir, f'sde_weight_{run_idx}.npy'), weight)
        # Concatenate reverse and forward traj, lnw, ts
        ts = torch.cat([ts_reverse.flip(0), ts_forward[1:]], dim=0)
        out = {
            "traj": sde_traj.detach().cpu().numpy(),  # (T,N,G)
            "lnw": traj_lnw.detach().cpu().numpy(),   # (T,N,1)
            "ts": ts.detach().cpu().numpy(),          # (T,)
            "point": sde_point_array,  # (T,N,G)
            "weight": weight,   # (T,N,1)
        }
        results.append(out)

        if output_dir is not None:
            np.savez(
                os.path.join(output_dir, f"{file_prefix}_run_{run_idx}.npz"),
                traj=out["traj"],
                lnw=out["lnw"],
                ts=out["ts"],
                point=out["point"],
                weight=out["weight"]
            )
    return results


def main():
    parser = argparse.ArgumentParser(description='Train DeepRUOT model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--num_runs', type=int, required=True, help='Path to config file')

    args = parser.parse_args()
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    config = load_and_merge_config(args.config)
    print(config)
    df, f_net, sf2m_score_model, exp_dir = load_data_model(config, device)
    dim = config['data']['dim']
    num_runs = args.num_runs
    sigma = config['score_train']['sigma']
    use_mass = config['use_mass']
    adata = generate_velocity(df, f_net, dim, exp_dir, device=device)
    # Trajectory / point samples for metrics
    if config['data']['hold_one_out'] & (config['data']['hold_out'] == 0):
        all_times = sorted(df.samples.unique())
        results_hold_start_out = generate_trajectories_sde_hold_start_out(df, f_net, sf2m_score_model, device, exp_dir, all_times, sigma, use_mass, num_points=None, num_runs = num_runs, output_dir = exp_dir)
    else:
        results = generate_trajectory_point(df, f_net, sf2m_score_model, dim, exp_dir, initial_num_points = None, sigma=sigma, use_mass = use_mass, device = device, traject_sample_number = 100, num_runs = num_runs, output_dir = exp_dir)
    print("Inference completed and results saved.")

if __name__ == '__main__':
    main() 


