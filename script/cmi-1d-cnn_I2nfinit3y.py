import os, json, joblib, numpy as np, pandas as pd
# os.environ['CUDA_VISIBLE_DEVICES'] = '5'
import random, math
from pathlib import Path
import warnings 
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, AdamW
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from timm.scheduler import CosineLRScheduler
from scipy.signal import firwin
import sys
sys.path.append("../metric")
from cmi_metric import CompetitionMetric
# 导入scipy的softmax
from scipy.special import softmax
from tqdm import tqdm
from copy import deepcopy
from IPython.display import clear_output
from functools import partial
from scipy.spatial.transform import Rotation as R
from types import SimpleNamespace
from pytorch_metric_learning import losses, miners

TRAIN = True                     
RAW_DIR = Path("../data")
EXPORT_DIR = Path("./")                                   
BATCH_SIZE = 64
PAD_PERCENTILE = 100
maxlen = PAD_PERCENTILE
LR_INIT = 1e-3
WD = 3e-3
MIXUP_ALPHA = 0.4
MASKING_PROB = 0.25
PATIENCE = 40
FOLDS = 5
random_state = 42
epochs_warmup = 20
warmup_lr_init = 1.822126131809773e-05
lr_min = 3.810323058740104e-09

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"▶ imports ready · pytorch {torch.__version__} · device: {device}")

# ================================
# Model Components
# ================================

class ImuFeatureExtractor(nn.Module):
    def __init__(self, fs=10., add_quaternion=False):
        super().__init__()
        self.fs = fs
        self.add_quaternion = add_quaternion

        k = 15
        
        self.lpf_acc   = nn.Conv1d(3, 3, k, padding=k//2, groups=3, bias=False)
        nn.init.kaiming_normal_(self.lpf_acc.weight, mode='fan_out')
        self.lpf_gyro = nn.Conv1d(3, 3, k, padding=k//2, groups=3, bias=False)
        nn.init.kaiming_normal_(self.lpf_gyro.weight, mode='fan_out')

    def forward(self, imu):
        # imu: 
        B, C, T = imu.shape
        acc  = imu[:, 0:3, :]                 # acc_x, acc_y, acc_z
        gyro = imu[:, 4:7, :]                 # gyro_x, gyro_y, gyro_z
        linear_acc      = imu[:, 7:10, :]
        angular_vel     = imu[:, 10:13, :]
        angular_distance = imu[:, 13:14, :] 
 

        # 线性加速度的幅度和jerk
        linear_acc_mag = torch.norm(linear_acc, dim=1, keepdim=True)
        linear_acc_mag_jerk = F.pad(linear_acc_mag[:, :, 1:] - linear_acc_mag[:, :, :-1], (1,0), 'replicate')  

        angular_vel_mag = torch.norm(angular_vel, dim=1, keepdim=True)
        angular_vel_mag_jerk = F.pad(angular_vel_mag[:, :, 1:] - angular_vel_mag[:, :, :-1], (1,0), 'replicate')  

        rot_angle = 2 * torch.acos(imu[:, 3, :].clamp(-1.0, 1.0)).unsqueeze(1) 
        rot_angle_vel = F.pad(rot_angle[:, :, 1:] - rot_angle[:, :, :-1], (1,0), 'replicate')

        # 1) magnitude
        acc_mag  = torch.norm(acc,  dim=1, keepdim=True)          # (B,1,T)
        gyro_mag = torch.norm(gyro, dim=1, keepdim=True)

        # 2) jerk 
        jerk = F.pad(acc[:, :, 1:] - acc[:, :, :-1], (1,0))       # (B,3,T)
        gyro_delta = F.pad(gyro[:, :, 1:] - gyro[:, :, :-1], (1,0))

        # 3) energy
        acc_pow  = acc ** 2
        gyro_pow = gyro ** 2

        # 4) LPF / HPF 
        acc_lpf  = self.lpf_acc(acc)
        acc_hpf  = acc - acc_lpf
        gyro_lpf = self.lpf_gyro(gyro)
        gyro_hpf = gyro - gyro_lpf

        acc_features = [
            acc, acc_mag,
            jerk, acc_pow,
            acc_lpf, acc_hpf,
            linear_acc, linear_acc_mag, linear_acc_mag_jerk,
        ]
        gyro_features = [
            gyro, gyro_mag,
            gyro_delta, gyro_pow,
            gyro_lpf, gyro_hpf,
            angular_vel, angular_vel_mag, angular_vel_mag_jerk, angular_distance,
            rot_angle, rot_angle_vel,
        ]
        # print(torch.cat(acc_features, dim=1).shape, torch.cat(gyro_features, dim=1).shape)
        features = acc_features + gyro_features
        return torch.cat(features, dim=1)

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1)
        return x * y.expand_as(x)

class ResidualSECNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_size=2, dropout=0.3):
        super().__init__()
        
        # First conv block
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        # Second conv block
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # SE block
        self.se = SEBlock(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        
        self.pool = nn.MaxPool1d(pool_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        shortcut = self.shortcut(x)
        
        # First conv
        out = F.silu(self.bn1(self.conv1(x)))
        # Second conv
        out = self.bn2(self.conv2(out))
        
        # SE block
        out = self.se(out)
        
        # Add shortcut
        out += shortcut
        out = F.silu(out)
        
        # Pool and dropout
        out = self.pool(out)
        out = self.dropout(out)
        
        return out

class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        # x shape: (batch, seq_len, hidden_dim)
        scores = torch.tanh(self.attention(x))  # (batch, seq_len, 1)
        weights = F.softmax(scores.squeeze(-1), dim=1)  # (batch, seq_len)
        context = torch.sum(x * weights.unsqueeze(-1), dim=1)  # (batch, hidden_dim)
        return context


class TwoBranchModel(nn.Module):
    def __init__(self, pad_len, imu_dim_raw, tof_dim, n_classes, dropouts=[0.3, 0.3, 0.3, 0.3, 0.4, 0.5, 0.3], 
                 feature_engineering=True, **kwargs):
        super().__init__()
        self.feature_engineering = feature_engineering
        if feature_engineering:
            self.imu_fe = ImuFeatureExtractor(**kwargs)
            imu_dim = 45
        else:
            self.imu_fe = nn.Identity()
            imu_dim = imu_dim_raw
            
        self.imu_dim = imu_dim
        self.tof_dim = tof_dim
        self.fir_nchan = imu_dim_raw
        

        # IMU deep branch
        self.acc_dim, self.rot_dim = 21, 24
        self.imu_block11 = ResidualSECNNBlock(self.acc_dim, 64, 3, 1, dropout=dropouts[0])
        self.imu_block12 = ResidualSECNNBlock(64, 128, 5, 1, dropout=dropouts[1])
        self.imu_block21 = ResidualSECNNBlock(self.rot_dim, 64, 3, 1, dropout=dropouts[0])
        self.imu_block22 = ResidualSECNNBlock(64, 128, 5, 1, dropout=dropouts[1])
        

        # v2
        self.tof_conv1 = ResidualSECNNBlock(tof_dim, 64, 3, 1, dropout=dropouts[2])
        self.tof_conv2 = ResidualSECNNBlock(64, 128, 3, 1, dropout=dropouts[3])

    

        # Gate
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dense1_gate = nn.Linear(pad_len, 16)
        self.dense2_gate = nn.Linear(16, 1)
        

        merged_channels = 256 + 128
        self.cnn_backbone1 = nn.Sequential(
            nn.Conv1d(merged_channels, 256, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(dropouts[4])
        )
        self.cnn_backbone2 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(dropouts[5])
        )

        

        # self.global_pool = AttentionLayer(512)
        self.global_pool = nn.AdaptiveAvgPool1d(1)  


        cnn_out_dim = 512
        self.dense1 = nn.Linear(cnn_out_dim, 256, bias=False)
        self.bn_dense1 = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(dropouts[5]) # 复用一个dropout值
        
        self.dense2 = nn.Linear(256, 128, bias=False)
        self.bn_dense2 = nn.BatchNorm1d(128)
        self.drop2 = nn.Dropout(dropouts[6])
        
        self.classifier = nn.Linear(128, n_classes)
        
    def forward(self, x):

        imu = x[:, :, :self.fir_nchan].transpose(1, 2)  # (B, D_imu_raw, T)
        tof = x[:, :, self.fir_nchan:].transpose(1, 2)  # (B, D_tof, T)

        imu  = self.imu_fe(imu)  # (B, D_imu, T)
        
        

        acc = imu[:, :self.acc_dim, :]  
        rot = imu[:, self.acc_dim:, :]  
        x11 = self.imu_block11(acc) # (B, 64, T)
        x11 = self.imu_block12(x11) # (B, 128, T)
        
        x12 = self.imu_block21(rot) # (B, 64, T)
        x12 = self.imu_block22(x12) # (B, 128, T)
        x1 = torch.cat([x11, x12], dim=1)

        
        # TOF
        
        # v2
        x2 = self.tof_conv1(tof)
        x2 = self.tof_conv2(x2)

        # Gate x2
        gate_input = self.pool(tof.transpose(1, 2)).squeeze(-1)
        gate_input = F.silu(self.dense1_gate(gate_input))
    
        gate = torch.sigmoid(self.dense2_gate(gate_input)) # -> (B, 1)
        x2 = x2 * gate.unsqueeze(-1)
        
        merged = torch.cat([x1, x2], dim=1) # (B, 256, T)
        

        cnn_out = self.cnn_backbone1(merged) # (B, 256, T)
        cnn_out = self.cnn_backbone2(cnn_out) # (B, 512, T)

        
        pooled = self.global_pool(cnn_out) # (B, 512, 1)
        pooled_flat = torch.flatten(pooled, 1) # (B, 512)

        
        x = F.silu(self.bn_dense1(self.dense1(pooled_flat)))
        x = self.drop1(x)
        x = F.silu(self.bn_dense2(self.dense2(x)))
        x = self.drop2(x)
        
        logits = self.classifier(x)
        return logits, x, gate


debug_model = TwoBranchModel(
    pad_len=100,
    imu_dim_raw=14,
    tof_dim=25,
    n_classes=18,
).to(device)

debug_x = torch.randn(4, 100, 14+25).to(device)  # (batch_size, channels, seq_len)
debug_model(debug_x)[0].shape



def remove_gravity_from_acc(acc_data, rot_data):

    if isinstance(acc_data, pd.DataFrame):
        acc_values = acc_data[['acc_x', 'acc_y', 'acc_z']].values
    else:
        acc_values = acc_data

    if isinstance(rot_data, pd.DataFrame):
        quat_values = rot_data[['rot_x', 'rot_y', 'rot_z', 'rot_w']].values
    else:
        quat_values = rot_data

    num_samples = acc_values.shape[0]
    linear_accel = np.zeros_like(acc_values)
    
    gravity_world = np.array([0, 0, 9.81])

    for i in range(num_samples):
        if np.all(np.isnan(quat_values[i])) or np.all(np.isclose(quat_values[i], 0)):
            linear_accel[i, :] = acc_values[i, :] 
            continue

        try:
            rotation = R.from_quat(quat_values[i])
            gravity_sensor_frame = rotation.apply(gravity_world, inverse=True)
            linear_accel[i, :] = acc_values[i, :] - gravity_sensor_frame
        except ValueError:
             linear_accel[i, :] = acc_values[i, :]
             
    return linear_accel

def calculate_angular_velocity_from_quat(rot_data, time_delta=1/200): # Assuming 200Hz sampling rate
    if isinstance(rot_data, pd.DataFrame):
        quat_values = rot_data[['rot_x', 'rot_y', 'rot_z', 'rot_w']].values
    else:
        quat_values = rot_data

    num_samples = quat_values.shape[0]
    angular_vel = np.zeros((num_samples, 3))

    for i in range(num_samples - 1):
        q_t = quat_values[i]
        q_t_plus_dt = quat_values[i+1]

        if np.all(np.isnan(q_t)) or np.all(np.isclose(q_t, 0)) or \
           np.all(np.isnan(q_t_plus_dt)) or np.all(np.isclose(q_t_plus_dt, 0)):
            continue

        try:
            rot_t = R.from_quat(q_t)
            rot_t_plus_dt = R.from_quat(q_t_plus_dt)

            # Calculate the relative rotation
            delta_rot = rot_t.inv() * rot_t_plus_dt
            
            # Convert delta rotation to angular velocity vector
            # The rotation vector (Euler axis * angle) scaled by 1/dt
            # is a good approximation for small delta_rot
            angular_vel[i, :] = delta_rot.as_rotvec() / time_delta
        except ValueError:
            # If quaternion is invalid, angular velocity remains zero
            pass
            
    return angular_vel

def calculate_angular_distance(rot_data):
    if isinstance(rot_data, pd.DataFrame):
        quat_values = rot_data[['rot_x', 'rot_y', 'rot_z', 'rot_w']].values
    else:
        quat_values = rot_data

    num_samples = quat_values.shape[0]
    angular_dist = np.zeros(num_samples)

    for i in range(num_samples - 1):
        q1 = quat_values[i]
        q2 = quat_values[i+1]

        if np.all(np.isnan(q1)) or np.all(np.isclose(q1, 0)) or \
           np.all(np.isnan(q2)) or np.all(np.isclose(q2, 0)):
            angular_dist[i] = 0 # Или np.nan, в зависимости от желаемого поведения
            continue
        try:
            # Преобразование кватернионов в объекты Rotation
            r1 = R.from_quat(q1)
            r2 = R.from_quat(q2)

            # Вычисление углового расстояния: 2 * arccos(|real(p * q*)|)
            # где p* - сопряженный кватернион q
            # В scipy.spatial.transform.Rotation, r1.inv() * r2 дает относительное вращение.
            # Угол этого относительного вращения - это и есть угловое расстояние.
            relative_rotation = r1.inv() * r2
            
            # Угол rotation vector соответствует угловому расстоянию
            # Норма rotation vector - это угол в радианах
            angle = np.linalg.norm(relative_rotation.as_rotvec())
            angular_dist[i] = angle
        except ValueError:
            angular_dist[i] = 0 # В случае недействительных кватернионов
            pass
            
    return angular_dist

class SignalTransform:
    def __init__(self, always_apply: bool = False, p: float = 0.5):
        self.always_apply = always_apply
        self.p = p

    def __call__(self, y: np.ndarray):
        if self.always_apply:
            return self.apply(y)
        else:
            if np.random.rand() < self.p:
                return self.apply(y)
            else:
                return y

    def apply(self, y: np.ndarray):
        raise NotImplementedError
        
class TimeStretch(SignalTransform):
    def __init__(self, max_rate=1.5, min_rate=0.5, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.max_rate = max_rate
        self.min_rate = min_rate
        self.always_apply = always_apply
        self.p = p

    def apply(self, x: np.ndarray):
        """
        Stretch a 1D or 2D array in time using linear interpolation.
        - x: np.ndarray of shape (L,) or (L, N)
        - rate: float, e.g., 1.2 for 20% longer, 0.8 for 20% shorter
        """
        rate = np.random.uniform(self.min_rate, self.max_rate)
        L = x.shape[0]
        L_new = int(L / rate)
        orig_idx = np.linspace(0, L - 1, num=L)
        new_idx = np.linspace(0, L - 1, num=L_new)

        if x.ndim == 1:
            return np.interp(new_idx, orig_idx, x)
        elif x.ndim == 2:
            return np.stack([
                np.interp(new_idx, orig_idx, x[:, i]) for i in range(x.shape[1])
            ], axis=1)
        else:
            raise ValueError("Only 1D or 2D arrays are supported.")
            
class TimeShift(SignalTransform):
    def __init__(self, always_apply=False, p=0.5, max_shift_pct=0.25, padding_mode="replace"):
        super().__init__(always_apply, p)
        
        assert 0 <= max_shift_pct <= 1.0, "`max_shift_pct` must be between 0 and 1"
        assert padding_mode in ["replace", "zero"], "`padding_mode` must be either 'replace' or 'zero'"
        
        self.max_shift_pct = max_shift_pct
        self.padding_mode = padding_mode

    def apply(self, x: np.ndarray, **params):
        assert x.ndim == 2, "`x` must be a 2D array with shape (L, N)"
        
        L = x.shape[0]
        max_shift = int(L * self.max_shift_pct)
        shift = np.random.randint(-max_shift, max_shift + 1)

        # Roll along time axis (axis=0)
        augmented = np.roll(x, shift, axis=0)

        if self.padding_mode == "zero":
            if shift > 0:
                augmented[:shift, :] = 0
            elif shift < 0:
                augmented[shift:, :] = 0

        return augmented


# ================================
# Data Handling
# ================================

def pad_sequences_torch(sequences, maxlen, padding='post', truncating='post', value=0.0):
    """PyTorch equivalent of Keras pad_sequences"""
    result = []
    for seq in sequences:
        if len(seq) >= maxlen:
            if truncating == 'post':
                seq = seq[:maxlen]
            else:  # 'pre'
                seq = seq[-maxlen:]
        else:
            pad_len = maxlen - len(seq)
            if padding == 'post':
                seq = np.concatenate([seq, np.full((pad_len, seq.shape[1]), value)])
            else:  # 'pre'
                seq = np.concatenate([np.full((pad_len, seq.shape[1]), value), seq])
        result.append(seq)
    return np.array(result, dtype=np.float32)

def preprocess_sequence(df_seq: pd.DataFrame, feature_cols: list, scaler: StandardScaler):
    """Normalizes and cleans the time series sequence"""
    mat = df_seq[feature_cols].ffill().bfill().fillna(0).values
    return scaler.transform(mat).astype('float32')

class CMI3Dataset(Dataset):
    def __init__(self,
                 X_list,
                 y_list,
                 maxlen,
                 mode="train",
                 imu_dim=14,
                 augment=None):
        self.X_list = X_list
        self.mode = mode
        self.y_list = y_list
        self.maxlen = maxlen
        self.imu_dim = imu_dim     
        self.augment = augment   

    def pad_sequences_torch(self, seq, maxlen, padding='post', truncating='post', value=0.0):

        if seq.shape[0] >= maxlen:
            if truncating == 'post':
                seq = seq[:maxlen]
            else:  # 'pre'
                seq = seq[-maxlen:]
        else:
            pad_len = maxlen - seq.shape[0]
            if padding == 'post':
                seq = np.concatenate([seq, np.full((pad_len, seq.shape[1]), value)])
            else:  # 'pre'
                seq = np.concatenate([np.full((pad_len, seq.shape[1]), value), seq])
        return seq  
        
    def __getitem__(self, index):
        X = self.X_list[index]
        y = self.y_list[index]

        # ---------- (A)  Augmentation ----------
        if self.mode == "train" and self.augment is not None:
            X = self.augment(X, self.imu_dim)     

        X = self.pad_sequences_torch(X, self.maxlen, 'pre', 'pre')
        return torch.from_numpy(X.copy()).float(), torch.from_numpy(y)
    
    def __len__(self):
        return len(self.X_list)


class EarlyStopping:
    """Early stopping utility"""
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()

class ModelEMA(nn.Module):
    def __init__(self, model, decay=0.99, device=None):
        super().__init__()
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)

def set_seed(seed: int = 42):
    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class Augment:
    def __init__(self,
                 p_jitter=0.8, sigma=0.02, scale_range=[0.9,1.1],
                 p_dropout=0.3,
                 p_moda=0.5,          
                 drift_std=0.005,     
                 drift_max=0.25,
                 ):      
        self.p_jitter  = p_jitter
        self.sigma     = sigma
        self.scale_min, self.scale_max = scale_range
        self.p_dropout = p_dropout
        self.p_moda    = p_moda
        self.drift_std = drift_std
        self.drift_max = drift_max

        self.time_stretch = TimeStretch(always_apply=False)
        self.time_shift = TimeShift(always_apply=False)
        
    # ---------- Jitter & Scaling ----------
    def jitter_scale(self, x: np.ndarray) -> np.ndarray:
        noise  = np.random.randn(*x.shape) * self.sigma
        scale  = np.random.uniform(self.scale_min,
                                   self.scale_max,
                                   size=(1, x.shape[1]))
        return (x + noise) * scale

    # ---------- Sensor Drop-out ----------
    def sensor_dropout(self,
                       x: np.ndarray,
                       imu_dim: int) -> np.ndarray:

        if random.random() < self.p_dropout:
            x[:, imu_dim:] = 0.0
        return x

    def motion_drift(self, x: np.ndarray, imu_dim: int) -> np.ndarray:

        T = x.shape[0]

        drift = np.cumsum(
            np.random.normal(scale=self.drift_std, size=(T, 1)),
            axis=0
        )
        drift = np.clip(drift, -self.drift_max, self.drift_max)   

        x[:, :6] += drift

        if imu_dim > 6:
            x[:, 6:imu_dim] += drift     
        return x
    
    
    # ---------- master call ----------
    def __call__(self,
                 x: np.ndarray,
                 imu_dim: int) -> np.ndarray:
            

        if random.random() < self.p_jitter:
            x = self.jitter_scale(x)

        if random.random() < self.p_moda:
            x = self.motion_drift(x, imu_dim)

        x = self.time_stretch(x)
        x = self.time_shift(x)
            
        x = self.sensor_dropout(x, imu_dim)
        return x

def augment_left_handed_sequence(seq_df: pd.DataFrame) -> pd.DataFrame:
    if seq_df['handedness'].iloc[0] == 0:
        seq_df = seq_df.copy()


        cols_3 = [c for c in seq_df.columns if any(p in c for p in ['tof_3', 'thm_3'])]
        cols_5 = [c for c in seq_df.columns if any(p in c for p in ['tof_5', 'thm_5'])]
        

        cols_3.sort()
        cols_5.sort()
        

        if len(cols_3) == len(cols_5) and len(cols_3) > 0:
            temp_data_3 = seq_df[cols_3].values
            seq_df[cols_3] = seq_df[cols_5].values
            seq_df[cols_5] = temp_data_3
            

        negate_cols = ['acc_x', 'rot_y', 'rot_z']
        existing_negate_cols = [col for col in negate_cols if col in seq_df.columns]
        if existing_negate_cols:
            seq_df[existing_negate_cols] *= -1
            
    return seq_df

set_seed(3407)

# %% [markdown]
# # Training

# %%
def mixup_collate_fn(batch, alpha, imu_dim, masking_prob=0.0):
    X_batch = torch.stack([item[0] for item in batch])
    y_batch = torch.stack([item[1] for item in batch])
    
    batch_size, seq_len, _ = X_batch.shape


    gate_target = torch.ones(batch_size, dtype=torch.float32)
    if masking_prob > 0:
        mask = torch.rand(batch_size) < masking_prob
        X_batch[mask, :, imu_dim:] = 0
        gate_target[mask] = 0.0


    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
        perm = torch.randperm(batch_size)

        if torch.rand(1) < 0.5:
            X_mix = lam * X_batch + (1 - lam) * X_batch[perm]
            y_mix = lam * y_batch + (1 - lam) * y_batch[perm]
            gate_target_mix = lam * gate_target + (1 - lam) * gate_target[perm]
        else:
            cut_ratio = np.sqrt(1. - lam)
            cut_len = int(seq_len * cut_ratio)
            center_x = np.random.randint(seq_len)
            start = np.clip(center_x - cut_len // 2, 0, seq_len)
            end = np.clip(center_x + cut_len // 2, 0, seq_len)
            lam = 1 - (end - start) / seq_len
            
            mask = torch.zeros_like(X_batch)
            mask[:, start:end, :] = 1.0
            X_mix = X_batch * (1. - mask) + X_batch[perm] * mask
            y_mix = y_batch * lam + y_batch[perm] * (1. - lam)
            gate_target_mix = lam * gate_target + (1 - lam) * gate_target[perm]
        
        return X_mix, y_mix, gate_target_mix

    return X_batch, y_batch, gate_target

# %%
if TRAIN:
    print("▶ TRAIN MODE – loading dataset …")
    df = pd.read_csv(RAW_DIR / "train.csv")
    df_demo = (pd.read_csv(RAW_DIR / "train_demographics.csv"))[['subject', 'handedness']]
    df = df.merge(df_demo, on='subject', how='left')

    df = df[~df['subject'].isin(["SUBJ_045235", "SUBJ_019262"])].reset_index(drop=True)

    print("  Transform the left hand into right hand")
    df = df.groupby('sequence_id', group_keys=False).apply(augment_left_handed_sequence)


    # Label encoding
    seq_gp = df.groupby('sequence_id')
    df_seq_id = pd.DataFrame()
    for seq_id, seq in tqdm(seq_gp):
        orientation = seq['orientation'].iloc[0]
        gesture = seq['gesture'].iloc[0]
        initial_behaviors = seq['behavior'].iloc[0]
        subject = seq['subject'].iloc[0]

        temp_df = pd.DataFrame({
            'seq_id': seq_id,
            'orientation': orientation,
            'gesture': gesture,
            'initial_behavior': initial_behaviors,
            'subject': subject
        }, index=[0])
        df_seq_id = pd.concat([df_seq_id, temp_df], ignore_index=True)
    df_seq_id['label'] = df_seq_id['orientation'] + '_' + df_seq_id['gesture'] + '_' + df_seq_id['initial_behavior']
    print(df_seq_id)
    print("unique behavior",df_seq_id['initial_behavior'].nunique())
    print("unique orientation",df_seq_id['orientation'].nunique())
    num_unique_labels = df_seq_id['label'].nunique()
    print(f"Number of unique labels: {num_unique_labels}")
    
    le = LabelEncoder()
    label_array = le.fit_transform(df_seq_id['label'])
    np.save(EXPORT_DIR / "gesture_classes.npy", le.classes_)

    # le = LabelEncoder()
    # df['gesture_int'] = le.fit_transform(df['gesture'])
    # np.save(EXPORT_DIR / "gesture_classes.npy", le.classes_)
    

    # Feature list
    meta_cols = {'gesture', 'gesture_int', 'sequence_type', 'behavior', 'orientation',
                 'row_id', 'subject', 'phase', 'sequence_id', 'sequence_counter', 'handedness'}
    feature_cols_meta = [c for c in df.columns if c not in meta_cols]


    print("  Removing gravity and calculating linear acceleration features...")

    linear_accel_list = []
    for _, group in df.groupby('sequence_id'):
        acc_data_group = group[['acc_x', 'acc_y', 'acc_z']]
        rot_data_group = group[['rot_x', 'rot_y', 'rot_z', 'rot_w']]
        linear_accel_group = remove_gravity_from_acc(acc_data_group, rot_data_group)
        linear_accel_list.append(pd.DataFrame(linear_accel_group, columns=['linear_acc_x', 'linear_acc_y', 'linear_acc_z'], index=group.index))
    df_linear_accel = pd.concat(linear_accel_list)

    df = pd.concat([df, df_linear_accel], axis=1)

    print("  Calculating angular velocity from quaternion derivatives...")
    angular_vel_list = []
    for _, group in df.groupby('sequence_id'):
        rot_data_group = group[['rot_x', 'rot_y', 'rot_z', 'rot_w']]
        angular_vel_group = calculate_angular_velocity_from_quat(rot_data_group)
        angular_vel_list.append(pd.DataFrame(angular_vel_group, columns=['angular_vel_x', 'angular_vel_y', 'angular_vel_z'], index=group.index))
    df_angular_vel = pd.concat(angular_vel_list)
   

    df = pd.concat([df, df_angular_vel], axis=1)

    print("  Calculating angular distance between successive quaternions...")
    angular_distance_list = []
    for _, group in df.groupby('sequence_id'):
        rot_data_group = group[['rot_x', 'rot_y', 'rot_z', 'rot_w']]
        angular_dist_group = calculate_angular_distance(rot_data_group)
        angular_distance_list.append(pd.DataFrame(angular_dist_group, columns=['angular_distance'], index=group.index))
    df_angular_distance = pd.concat(angular_distance_list)
    

    df = pd.concat([df, df_angular_distance], axis=1)


    imu_cols_base = [c for c in feature_cols_meta if not (c.startswith('thm_') or c.startswith('tof_'))]
    imu_engineered_features = [
    'linear_acc_x', 'linear_acc_y', 'linear_acc_z',
    'angular_vel_x', 'angular_vel_y', 'angular_vel_z', 'angular_distance',
    ]
    imu_cols = imu_cols_base + imu_engineered_features


    thm_cols_original = [c for c in df.columns if c.startswith('thm_')]
    tof_aggregated_cols_template = []
    stat_types = ['mean', 'std', 'min', 'max']
    for i in range(1, 6):
        tof_aggregated_cols_template.extend([f'tof_{i}_{stat}' for stat in stat_types])
    # for i in range(1, 6):
    #     tof_aggregated_cols_template.extend([f'tof_{i}_v{p}' for p in range(64)])

    tof_cols = thm_cols_original + tof_aggregated_cols_template
    feature_cols = imu_cols + tof_cols
    print(f"  IMU {len(imu_cols)} | TOF/THM {len(tof_cols)} | total {len(feature_cols)} features")

    
    # Build sequences
    seq_gp = df.groupby('sequence_id')
    all_steps_for_scaler_list = []
    X_list, y_list, id_list, hand_list, lens = [], [], [], [],[]
    for idx, (seq_id, seq) in tqdm(enumerate(seq_gp)):
        seq_df = seq.copy()
        for i in range(1, 6):
            pixel_cols_tof = [f"tof_{i}_v{p}" for p in range(64)]
            tof_sensor_data = seq_df[pixel_cols_tof].replace(-1, np.nan)
            seq_df[f'tof_{i}_mean'] = tof_sensor_data.mean(axis=1)
            seq_df[f'tof_{i}_std']  = tof_sensor_data.std(axis=1)
            seq_df[f'tof_{i}_min']  = tof_sensor_data.min(axis=1)
            seq_df[f'tof_{i}_max']  = tof_sensor_data.max(axis=1)
           

        mat = seq_df[feature_cols].ffill().bfill().fillna(0).values.astype('float32')
        all_steps_for_scaler_list.append(mat)
        X_list.append(mat)
        # y_list.append(seq['gesture_int'].iloc[0].astype(np.int32))
        y_list.append(label_array[idx])
        hand_list.append(seq['handedness'].iloc[0])
        id_list.append(seq_id)
        lens.append(len(mat))

    # Global scaler
    print("  Fitting StandardScaler...")
    all_steps_concatenated = np.concatenate(all_steps_for_scaler_list, axis=0)
    scaler = StandardScaler().fit(all_steps_concatenated)
    joblib.dump(scaler, EXPORT_DIR / "scaler.pkl")
    del all_steps_for_scaler_list, all_steps_concatenated

    print("  Scaling and padding sequences...")
    X_list = [scaler.transform(x_seq) for x_seq in X_list]
    
    pad_len = PAD_PERCENTILE#int(np.percentile(lens, PAD_PERCENTILE))
    print(pad_len)
    np.save(EXPORT_DIR / "sequence_maxlen.npy", pad_len)
    np.save(EXPORT_DIR / "feature_cols.npy", np.array(feature_cols))
    id_list = np.array(id_list)
    hand_list = np.array(hand_list)
    X_list_all = pad_sequences_torch(X_list, maxlen=pad_len, padding='pre', truncating='pre')
    y_list_all = np.eye(len(le.classes_))[y_list].astype(np.float32)  # One-hot encoding

    augmenter = Augment(
        p_jitter=0.9844818619033621, sigma=0.03291295776089293, scale_range=(0.7542342630597011,1.1625052821731077),
        p_dropout=0.41782786013520684,
        p_moda=0.3910622476959722, drift_std=0.0040285239353308015, drift_max=0.3929358950258158    
    )

# %%
metric_loss_fn = losses.TripletMarginLoss(margin=0.2) 
miner = miners.TripletMarginMiner(margin=0.2, type_of_triplets="hard")

# %%
EPOCHS = 160


groups_by_subject = seq_gp['subject'].first()

oof = np.zeros((X_list_all.shape[0], ), dtype=np.float32)


collate_fn_with_args = partial(mixup_collate_fn, 
                               alpha=MIXUP_ALPHA, 
                               imu_dim=len(imu_cols), 
                               masking_prob=MASKING_PROB,
                               )
skf = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(skf.split(id_list, seq_gp['sequence_type'].first(), groups=groups_by_subject)):
        
    best_h_f1 = 0
    train_list= X_list_all[train_idx]
    train_y_list= y_list_all[train_idx]
    val_list = X_list_all[val_idx]
    val_y_list= y_list_all[val_idx]


    
    
    # Data loaders
    train_dataset = CMI3Dataset(train_list, train_y_list, maxlen, mode="train", imu_dim=len(imu_cols),
                            augment=augmenter)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_with_args, num_workers=4,drop_last=True)

    val_dataset = CMI3Dataset(val_list, val_y_list, maxlen, mode="val")
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,drop_last=False)


    # Model
    model = TwoBranchModel(maxlen, len(imu_cols), len(tof_cols), 
                    len(le.classes_)).to(device)
    ema = ModelEMA(model, decay=0.99)
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=LR_INIT, weight_decay=WD)
    steps_per_epoch = len(train_loader)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5*steps_per_epoch)
    
    steps_per_epoch = len(train_loader)
    nbatch = len(train_loader)
    warmup = epochs_warmup * nbatch
    nsteps = EPOCHS * nbatch
    scheduler = CosineLRScheduler(optimizer,
                        warmup_t=warmup, warmup_lr_init=warmup_lr_init, warmup_prefix=True,
                        t_initial=(nsteps - warmup), lr_min=lr_min) 

    early_stopping = EarlyStopping(patience=PATIENCE, restore_best_weights=True)

    train_loss = 0.0
    train_acc = 0.0
    val_loss = 0.0
    val_acc = 0.0
    val_best_acc = 0.0
    i_scheduler = 0
    
    # Training loop
    print("▶ Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        train_preds = []
        train_targets = []

        if epoch == 10:
            ema.set(model)  
        for X, y, gate_target in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):  
            X, y, gate_target = X.float().to(device), y.to(device),\
                                                            gate_target.to(device)

            optimizer.zero_grad()
            logits, embedding, gate_pred = model(X)  # logits: (B, n_classes), gate_pred: (B, 1)

            classification_loss = -torch.sum(F.log_softmax(logits, dim=1) * y, dim=1).mean()
            hard_triplets = miner(embedding, y.argmax(dim=1))
            metric_loss = metric_loss_fn(embedding, y.argmax(dim=1), hard_triplets)
            loss = classification_loss + metric_loss+\
                                0.2*F.binary_cross_entropy(gate_pred.squeeze(-1), gate_target)

            loss.backward()
            optimizer.step()
            if epoch >= 10:
                ema.update(model)
            train_preds.extend(logits.argmax(dim=1).cpu().numpy())
            train_targets.extend(y.argmax(dim=1).cpu().numpy())
            scheduler.step(i_scheduler)
            i_scheduler +=1

            train_loss += loss.item()
            
        model.eval()
        with torch.inference_mode():
            val_preds = []
            val_targets = []
            for X, y in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
                half = BATCH_SIZE // 2         

                X, y = X.float().to(device), y.to(device)
                
                if epoch >= 10:
                    logits = ema.module(X)[0]
                else:
                    logits = model(X)[0]
                # logits = logits.mean(dim=1) 
                val_preds.extend(logits.argmax(dim=1).cpu().numpy())
                val_targets.extend(y.argmax(dim=1).cpu().numpy())
                
                loss = F.cross_entropy(logits, y)
                # loss = loss_fn(logits, y)
                val_loss += loss.item()

        oof[val_idx] = val_preds

        train_pred_true = [x.split('_')[1] for x in le.classes_[train_preds]]
        val_pred_true = [x.split('_')[1] for x in le.classes_[val_preds]]
        train_targets_true = [x.split('_')[1] for x in le.classes_[train_targets]]
        val_targets_true = [x.split('_')[1] for x in le.classes_[val_targets]]

        train_acc = CompetitionMetric().calculate_hierarchical_f1(
            pd.DataFrame({'gesture': train_targets_true}),
            pd.DataFrame({'gesture': train_pred_true}))
        val_acc = CompetitionMetric().calculate_hierarchical_f1(
            pd.DataFrame({'gesture': val_targets_true}),
            pd.DataFrame({'gesture': val_pred_true}))
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1}/{EPOCHS} -> Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Train H-F1: {train_acc:.4f} | Valid H-F1: {val_acc:.4f}")
    

        if val_acc > best_h_f1:
            best_h_f1 = val_acc
            
            torch.save({
            'model_state_dict': ema.module.state_dict(),
            'imu_dim': len(imu_cols),
            'tof_dim': len(tof_cols),
            'n_classes': len(le.classes_),
            'pad_len': pad_len
            }, EXPORT_DIR / f"gesture_two_branch_fold{fold}.pth")
            print(f"  New best model saved with H-F1: {best_h_f1:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{PATIENCE}")

        if patience_counter >= PATIENCE:
            print("  Early stopping triggered.")
            break

        print("\n")

      
    clear_output(wait=True) 
    print(f"fold: {fold} val_all_acc: {best_h_f1:.4f}")
    print("✔ Training done – artefacts saved in", EXPORT_DIR)

# %%
val_pred_true = le.classes_[val_preds.argmax(1)]
val_pred_true = [x.split('_')[1] for x in val_pred_true]
val_target_true = [x.split('_')[1] for x in le.classes_[val_targets]]

# %%
CompetitionMetric().calculate_hierarchical_f1(
            pd.DataFrame({'gesture': val_target_true}),
            pd.DataFrame({'gesture': val_pred_true}))

# %%

groups_by_subject = seq_gp['subject'].first()

skf = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=42)

oof = np.zeros((X_list_all.shape[0], ), dtype=np.float32)
oof_imu = np.zeros((X_list_all.shape[0], ))
for fold, (train_idx, val_idx) in enumerate(skf.split(id_list, seq_gp['sequence_type'].first(), groups=groups_by_subject)):
    _, val_list = X_list_all[train_idx], X_list_all[val_idx]
    _, val_y_list = y_list_all[train_idx], y_list_all[val_idx]

    val_dataset = CMI3Dataset(val_list, val_y_list, maxlen, mode="val")
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,drop_last=False)
    

    # best_model_path = Path("/kaggle/input/cmi-clear-imu-only") / f"exp172_gesture_two_branch_fold{fold}.pth"
    best_model_path = EXPORT_DIR/ f"gesture_two_branch_fold{fold}.pth"
    print(f"Evaluating fold {fold+1} with model {best_model_path}")
    
    eval_model = TwoBranchModel(maxlen, len(imu_cols), len(tof_cols), 
                    len(le.classes_)).to(device)
    eval_model.load_state_dict(torch.load(best_model_path, map_location=device)['model_state_dict'])
    eval_model.to(device)
    eval_model.eval()
    with torch.inference_mode():
        val_preds = []
        val_targets = []
        val_preds2 = []
        for X, y in tqdm(val_loader, desc=f"[Val]"):
            logits2 = eval_model(X.float().to(device))[0]
            val_preds2.extend(logits2.argmax(dim=1).cpu().numpy())
        
            half = BATCH_SIZE // 2         

            x_front = X[:half]               
            x_back  = X[half:].clone()      
            
            x_back[:, :, len(imu_cols):] = 0.0    
            X = torch.cat([x_front, x_back], dim=0)  # (B, C, T)
            X, y = X.float().to(device), y.to(device)
            
            logits = eval_model(X)[0]
            val_preds.extend(logits.argmax(dim=1).cpu().numpy())
            val_targets.extend(y.argmax(dim=1).cpu().numpy())

    oof_imu[val_idx] = val_preds
    oof[val_idx] = val_preds2

    val_pred_true = le.classes_[val_preds]
    val_pred_true = [x.split('_')[1] for x in val_pred_true]

    val_pred2_true = [x.split('_')[1] for x in le.classes_[val_preds2]]
    val_target_true = [x.split('_')[1] for x in le.classes_[val_targets]]

    h_f1 = CompetitionMetric().calculate_hierarchical_f1(
        pd.DataFrame({'gesture': val_target_true}),
        pd.DataFrame({'gesture': val_pred_true})
    )

    all_h_f1 = CompetitionMetric().calculate_hierarchical_f1(
        pd.DataFrame({'gesture': val_target_true}),
        pd.DataFrame({'gesture': val_pred2_true})
    )

    print(f"Fold {fold+1} H-F1 = {h_f1:.4f}")
    print(f"Fold {fold+1} All H-F1 = {all_h_f1:.4f}")
    print("\n")

oof_imu_true = [x.split('_')[1] for x in le.classes_[oof_imu.astype(int)]]
oof_true = [x.split('_')[1] for x in le.classes_[oof.astype(int)]]

y_list_all_true = [x.split('_')[1] for x in le.classes_[y_list_all.argmax(axis=1)]]

imu_h_f1 = CompetitionMetric().calculate_hierarchical_f1(
    pd.DataFrame({'gesture': oof_imu_true}),
    pd.DataFrame({'gesture': y_list_all_true})
)

h_f1 = CompetitionMetric().calculate_hierarchical_f1(
    pd.DataFrame({'gesture': oof_true}),
    pd.DataFrame({'gesture': y_list_all_true})
)

print(f"OOF H-F1 = {h_f1:.4f}")
print(f"OOF IMU H-F1 = {imu_h_f1:.4f}")

# %%
(imu_h_f1 + h_f1) / 2

# %%
trian_dem = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv")
subject_handedness = trian_dem.set_index('subject')['handedness'].to_dict()


handedness_filtered = groups_by_subject.map(subject_handedness)
handedness_filtered = handedness_filtered.reset_index(drop=True)
left_filtered_idx = handedness_filtered[handedness_filtered == 0].index
right_filtered_idx = handedness_filtered[handedness_filtered == 1].index


oof_left = oof[left_filtered_idx]
oof_imu_left = oof_imu[left_filtered_idx]
oof_right = oof[right_filtered_idx]
oof_imu_right = oof_imu[right_filtered_idx]

left_h_f1 = CompetitionMetric().calculate_hierarchical_f1(
    pd.DataFrame({'gesture': le.classes_[oof_left.astype(int)]}),
    pd.DataFrame({'gesture': le.classes_[y_list_all[left_filtered_idx].argmax(axis=1)]})
)
left_imu_h_f1 = CompetitionMetric().calculate_hierarchical_f1(
    pd.DataFrame({'gesture': le.classes_[oof_imu_left.astype(int)]}),
    pd.DataFrame({'gesture': le.classes_[y_list_all[left_filtered_idx].argmax(axis=1)]})
)

right_h_f1 = CompetitionMetric().calculate_hierarchical_f1(
    pd.DataFrame({'gesture': le.classes_[oof_right.astype(int)]}),
    pd.DataFrame({'gesture': le.classes_[y_list_all[right_filtered_idx].argmax(axis=1)]})
)

right_imu_h_f1 = CompetitionMetric().calculate_hierarchical_f1(
    pd.DataFrame({'gesture': le.classes_[oof_imu_right.astype(int)]}),
    pd.DataFrame({'gesture': le.classes_[y_list_all[right_filtered_idx].argmax(axis=1)]})
)

print(f"Right Handed OOF H-F1 = {right_h_f1:.4f}")
print(f"Right Handed OOF IMU H-F1 = {right_imu_h_f1:.4f}")
print(f"Right Handed OOF H-F1 + IMU H-F1 = {(right_h_f1 + right_imu_h_f1) / 2:.4f}")

print("\n")

print(f"Left Handed OOF H-F1 = {left_h_f1:.4f}")
print(f"Left Handed OOF IMU H-F1 = {left_imu_h_f1:.4f}")
print(f"Left Handed OOF H-F1 + IMU H-F1 = {(left_h_f1 + left_imu_h_f1) / 2:.4f}")


# Evaluating fold 1 with model gesture_two_branch_fold0.pth
# [Val]: 100%|██████████| 26/26 [00:00<00:00, 63.48it/s]
# Fold 1 H-F1 = 0.8224
# Fold 1 All H-F1 = 0.8599

# Evaluating fold 2 with model gesture_two_branch_fold1.pth
# [Val]: 100%|██████████| 26/26 [00:00<00:00, 62.77it/s]
# Fold 2 H-F1 = 0.8441
# Fold 2 All H-F1 = 0.8615

# Evaluating fold 3 with model gesture_two_branch_fold2.pth
# [Val]: 100%|██████████| 26/26 [00:00<00:00, 61.91it/s]
# Fold 3 H-F1 = 0.8618
# Fold 3 All H-F1 = 0.8967

# Evaluating fold 4 with model gesture_two_branch_fold3.pth
# [Val]: 100%|██████████| 26/26 [00:00<00:00, 63.34it/s]
# Fold 4 H-F1 = 0.8748
# Fold 4 All H-F1 = 0.8964

# Evaluating fold 5 with model gesture_two_branch_fold4.pth
# [Val]: 100%|██████████| 23/23 [00:00<00:00, 63.52it/s]
# Fold 5 H-F1 = 0.8687
# Fold 5 All H-F1 = 0.8861

# OOF H-F1 = 0.8800
# OOF IMU H-F1 = 0.8542

# Right Handed OOF H-F1 = 0.8840
# Right Handed OOF IMU H-F1 = 0.8591
# Right Handed OOF H-F1 + IMU H-F1 = 0.8716


# Left Handed OOF H-F1 = 0.8527
# Left Handed OOF IMU H-F1 = 0.8209
# Left Handed OOF H-F1 + IMU H-F1 = 0.8368