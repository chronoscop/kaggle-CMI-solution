import os, json, joblib, numpy as np, pandas as pd
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from timm.scheduler import CosineLRScheduler
from scipy.signal import firwin
import sys
sys.path.append("../metric")
from cmi_metric import CompetitionMetric
from tqdm import tqdm
from copy import deepcopy
from IPython.display import clear_output
from functools import partial
from scipy.spatial.transform import Rotation as R
from types import SimpleNamespace
from pytorch_metric_learning import losses, miners


# Configuration
TRAIN = True                     
RAW_DIR = Path("../data")
EXPORT_DIR = Path("../models")                                   
BATCH_SIZE = 64
PAD_PERCENTILE = 100
maxlen = PAD_PERCENTILE
LR_INIT = 1e-3
WD = 3e-3
MIXUP_ALPHA = 0.4
MASKING_PROB = 0.25
PATIENCE = 30
FOLDS = 10
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
            rot_angle, rot_angle_vel
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

class FuseGate(nn.Module):
    def __init__(self, pad_len, imu_dim_raw, tof_dim, n_classes, dropouts=[0.25, 0.25, 0.25, 0.25, 0.32, 0.4, 0.3], 
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
        # self.imu_block1 = ResidualSECNNBlock(imu_dim, 64, 3, 1, dropout=dropouts[0], weight_decay=weight_decay)
        # self.imu_block2 = ResidualSECNNBlock(64, 128, 5, 1, dropout=dropouts[1], weight_decay=weight_decay)
        self.acc_dim, self.rot_dim = 21, 24
        self.imu_block11 = ResidualSECNNBlock(self.acc_dim, 64, 3, 1, dropout=dropouts[0])
        self.imu_block12 = ResidualSECNNBlock(64, 128, 5, 1, dropout=dropouts[1])
        self.imu_block21 = ResidualSECNNBlock(self.rot_dim, 64, 3, 1, dropout=dropouts[0])
        self.imu_block22 = ResidualSECNNBlock(64, 128, 5, 1, dropout=dropouts[1])
        
        # TOF/Thermal lighter branch
        self.tof_conv1 = nn.Conv1d(tof_dim, 64, 3, padding=1, bias=False)
        self.tof_bn1 = nn.BatchNorm1d(64)
        self.tof_drop1 = nn.Dropout(dropouts[2])
        
        self.tof_conv2 = nn.Conv1d(64, 128, 3, padding=1, bias=False)
        self.tof_bn2 = nn.BatchNorm1d(128)
        self.tof_drop2 = nn.Dropout(dropouts[3])

        # Gate
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dense1_gate = nn.Linear(pad_len, 16)
        self.dense2_gate = nn.Linear(16, 1)
        

        merged_channels = 256+ 128
        self.cnn_backbone1 = nn.Sequential(
            nn.Conv1d(merged_channels, 256, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(dropouts[4])
        )
        self.cnn_backbone2 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(dropouts[5])
        )
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.lstm_backbone1 = nn.LSTM(
            input_size=512,
            hidden_size=256,
            num_layers=1,  
            batch_first=True,
            bidirectional=True
        )
        self.lstm_layer_norm1 = nn.LayerNorm(256)  
        self.lstm_leaky_relu1 = nn.LeakyReLU()
        self.lstm_dropout1 = nn.Dropout(dropouts[4])
        
        self.attn_fc_lstm2 = nn.Linear(512, 1) 
        
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

        imu = self.imu_fe(imu)  # (B, D_imu, T)
        
        # IMU 分支
        # x1 = self.imu_block1(imu) # (B, 64, T)
        # x1 = self.imu_block2(x1) # (B, 128, T)
        acc = imu[:, :self.acc_dim, :]  # 保留前21个通道作为加速度特征
        rot = imu[:, self.acc_dim:, :]  
        x11 = self.imu_block11(acc) # (B, 64, T)
        x11 = self.imu_block12(x11) # (B, 64, T)
        
        x12 = self.imu_block21(rot) # (B, 64, T)
        x12 = self.imu_block22(x12) # (B, 64, T)
        x1 = torch.cat([x11, x12], dim=1)
        
        # TOF 分支
        x2 = F.leaky_relu(self.tof_bn1(self.tof_conv1(tof)))
        x2 = self.tof_drop1(x2) # (B, 64, T)
        x2 = F.leaky_relu(self.tof_bn2(self.tof_conv2(x2)))
        x2 = self.tof_drop2(x2) # (B, 128, T)

        # Gate x2
        gate_input = self.pool(tof.transpose(1, 2)).squeeze(-1)
        gate_input = F.relu(self.dense1_gate(gate_input))
    
        gate = torch.sigmoid(self.dense2_gate(gate_input)) # -> (B, 1)
        x2 = x2 * gate.unsqueeze(-1)
        

        merged = torch.cat([x1, x2], dim=1) # (B, 256, T)

        cnn_feat = self.cnn_backbone1(merged) # (B, 256, T)
        cnn_feat = self.cnn_backbone2(cnn_feat) # (B, 512, T)
        # cnn_out_final = self.global_pool(cnn_feat).squeeze(-1)  # (B, 512, T) -> (B, 512, 1) -> (B, 512)
        
        # lstm_feat = merged.permute(0, 2, 1)  # (B, T, 256)
        lstm_feat = cnn_feat.permute(0, 2, 1)
        lstm_out, _ = self.lstm_backbone1(lstm_feat)  # (B, T, 512)
    
        # GRU注意力机制
        attn_scores_lstm = torch.softmax(self.attn_fc_lstm2(lstm_out), dim=1)  # (B, T, 1)
        lstm_out_final = torch.sum(attn_scores_lstm * lstm_out, dim=1)

        # Element-wise fusion（必须维度一致）
        pooled_flat = lstm_out_final
        
        # pooled_flat = torch.sum(attn_scores * lstm_out, dim=1) 
        # pooled = self.global_pool(cnn_out) # (B, 512, 1)
        # pooled_flat = torch.flatten(pooled, 1) # (B, 512)
        
        x = F.leaky_relu(self.bn_dense1(self.dense1(pooled_flat)))
        x = self.drop1(x)
        x = F.leaky_relu(self.bn_dense2(self.dense2(x)))
        x = self.drop2(x)
        
        logits = self.classifier(x)
        return logits, x,gate

debug_model = FuseGate(
    pad_len=100,
    imu_dim_raw=32,  
    tof_dim=78,     
    n_classes=18,   
).to(device)

debug_x = torch.randn(4, 100, 32+78).to(device)  # (batch_size, channels, seq_len)
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

def calculate_tof_optical_flow(df):
    flow_features_list = []
    for _, group in df.groupby('sequence_id'):
        flow_features = pd.DataFrame(index=group.index)
        
        # 只选择关键传感器 (1, 3, 5) - 减少传感器数量
        for i in [1, 3, 5]:
            cols = [f'tof_{i}_v{j}' for j in range(64)]
            images = group[cols].values.reshape(-1, 8, 8)
            
            # 计算相邻帧差（简化光流）
            if len(images) > 1:
                flow = np.diff(images, axis=0)
                flow = np.pad(flow, ((0,1), (0,0), (0,0)), mode='constant')
            else:
                flow = np.zeros_like(images)
            
            # 提取关键光流特征
            flow_features[f'tof_{i}_flow_mean'] = flow.mean(axis=(1,2))
            flow_features[f'tof_{i}_motion_energy'] = (flow**2).mean(axis=(1,2))
        
        # 跨传感器统计特征 (4个)
        sensor_flows = [f'tof_{i}_flow_mean' for i in [1, 3, 5]]
        flow_features['tof_flow_mean_all'] = flow_features[sensor_flows].mean(axis=1)
        flow_features['tof_flow_std_all'] = flow_features[sensor_flows].std(axis=1)
        flow_features['tof_motion_max'] = flow_features[[f'tof_{i}_motion_energy' for i in [1, 3, 5]]].max(axis=1)
        flow_features['tof_motion_sum'] = flow_features[[f'tof_{i}_motion_energy' for i in [1, 3, 5]]].sum(axis=1)
        
        flow_features_list.append(flow_features)
    return pd.concat(flow_features_list)

def calculate_tof_3d_features(df):
    geometric_features_list = []
    
    for _, group in df.groupby('sequence_id'):
        geometric_features = pd.DataFrame(index=group.index)
        
        # 只选择关键传感器 (2, 4) - 减少传感器数量
        for i in [2, 4]:
            cols = [f'tof_{i}_v{j}' for j in range(64)]
            depths = group[cols].values
            
            # 核心几何特征
            geometric_features[f'tof_{i}_depth_mean'] = depths.mean(axis=1)
            geometric_features[f'tof_{i}_depth_std'] = depths.std(axis=1)
            geometric_features[f'tof_{i}_depth_range'] = depths.max(axis=1) - depths.min(axis=1)
            
            # 表面粗糙度特征
            curvature = []
            for depth_map in depths:
                depth_map = depth_map.reshape(8, 8)
                # 简化的表面变化度量
                grad_x = np.gradient(depth_map, axis=1)
                grad_y = np.gradient(depth_map, axis=0)
                surface_variation = np.sqrt(grad_x**2 + grad_y**2).mean()
                curvature.append(surface_variation)
            geometric_features[f'tof_{i}_surface_roughness'] = curvature
        
        # 跨传感器组合特征 (2个)
        geometric_features['tof_depth_mean_diff'] = (
            geometric_features['tof_2_depth_mean'] - geometric_features['tof_4_depth_mean']
        )
        geometric_features['tof_depth_std_sum'] = (
            geometric_features['tof_2_depth_std'] + geometric_features['tof_4_depth_std']
        )
        
        geometric_features_list.append(geometric_features)
    return pd.concat(geometric_features_list)

CENTER_XY = 3.5
y_coords, x_coords = np.mgrid[0:8, 0:8]
def calculate_tof_centroid_dist(row: pd.Series) -> float:
    """
    计算单行数据（一个时间点）的tof_centroid_dist
    """
    all_centroids = []
    all_energies = []

    # 第A步：为5个传感器分别计算质心
    for i in range(1, 6):
        # 提取、处理、重塑
        tof_cols = [f'tof_{i}_v{j}' for j in range(64)]
        pixel_values = row[tof_cols].values.astype(float)
        pixel_values[pixel_values == -1] = 0
        grid = pixel_values.reshape(8, 8)
        
        total_energy = np.sum(grid)

        if total_energy < 1e-6: # 如果没有信号
            cx, cy = CENTER_XY, CENTER_XY
        else:
            # 计算质心
            cx = np.sum(x_coords * grid) / total_energy
            cy = np.sum(y_coords * grid) / total_energy
        
        all_centroids.append([cx, cy])
        all_energies.append(total_energy)

    # 第B步：计算总体质心（加权平均）
    all_centroids = np.array(all_centroids)
    all_energies = np.array(all_energies)
    total_system_energy = np.sum(all_energies)

    if total_system_energy < 1e-6:
        overall_cx, overall_cy = CENTER_XY, CENTER_XY
    else:
        overall_cx = np.sum(all_centroids[:, 0] * all_energies) / total_system_energy
        overall_cy = np.sum(all_centroids[:, 1] * all_energies) / total_system_energy

    # 第C步：计算到中心的距离
    dist = np.sqrt((overall_cx - CENTER_XY)**2 + (overall_cy - CENTER_XY)**2)
    
    return dist

def compute_global_acceleration_final(df, chunk_size=1000):
    df['acc_global_x'] = np.nan
    df['acc_global_y'] = np.nan
    df['acc_global_z'] = np.nan
    # 分块处理
    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i+chunk_size].copy()
        
        # 1. 提取数据并检查NaN
        acc_local = chunk[['acc_x', 'acc_y', 'acc_z']].values
        quat = chunk[['rot_w', 'rot_x', 'rot_y', 'rot_z']].values
        
        # 2. 处理NaN值
        nan_mask = np.isnan(quat).any(axis=1)
        if np.any(nan_mask):
            # 用默认单位四元数替换NaN
            quat[nan_mask] = [1.0, 0.0, 0.0, 0.0]
        
        # 3. 处理零模四元数
        norms = np.linalg.norm(quat, axis=1)
        zero_mask = norms < 1e-10
        if np.any(zero_mask):
            quat[zero_mask] = [1.0, 0.0, 0.0, 0.0]
        
        # 4. 归一化四元数
        norms = np.linalg.norm(quat, axis=1, keepdims=True)
        safe_quat = quat / norms
        
        # 5. 转换为旋转矩阵
        quat_adjusted = np.roll(safe_quat, shift=-1, axis=-1)
        rotations = R.from_quat(quat_adjusted)
        rotation_matrices = rotations.as_matrix()  
        # 6. 计算全局加速度
        acc_global = np.einsum('tij,tj->ti', rotation_matrices, acc_local)
        # 7. 存储结果
        df.iloc[i:i+chunk_size, df.columns.get_loc('acc_global_x')] = acc_global[:, 0]
        df.iloc[i:i+chunk_size, df.columns.get_loc('acc_global_y')] = acc_global[:, 1]
        df.iloc[i:i+chunk_size, df.columns.get_loc('acc_global_z')] = acc_global[:, 2]
    return df

def bin_imu_features_for_bfrb(df):
    binned_features = pd.DataFrame(index=df.index)
    
    # 1. 动作强度分箱（BFRB通常更用力）
    accel_mag = np.sqrt(df['linear_acc_x']**2 + df['linear_acc_y']**2 + df['linear_acc_z']**2)
    # 根据文献：BFRB行为强度通常更高
    binned_features['accel_intensity'] = pd.cut(accel_mag, 
                                               bins=[0, 0.5, 2.0, 5.0, np.inf],
                                               labels=[0, 1, 2, 3])  # 0=静止, 3=剧烈
    

    if 'angvel_mag' in df.columns:
        binned_features['rotation_intensity'] = pd.cut(df['angvel_mag'],
                                                      bins=[0, 0.1, 0.5, 1.5, np.inf],
                                                      labels=[0, 1, 2, 3])
    

    ax, ay, az = df['linear_acc_x'], df['linear_acc_y'], df['linear_acc_z']
    # 简化为主要运动方向
    directions = []
    for i in range(len(df)):
        x, y, z = abs(ax.iloc[i]), abs(ay.iloc[i]), abs(az.iloc[i])
        if max(x, y, z) == x:
            directions.append(0)  # X方向主导
        elif max(x, y, z) == y:
            directions.append(1)  # Y方向主导
        else:
            directions.append(2)  # Z方向主导
    binned_features['motion_direction'] = directions
    
    return binned_features

def calculate_attitude_angles(quat_data):
    """
    从四元数计算姿态角（roll, pitch, yaw）
    
    参数:
        quat_data (np.ndarray): 四元数数组，形状为 (N, 4)，列顺序为 [rot_x, rot_y, rot_z, rot_w]
                               注意：这里假设输入顺序是x,y,z,w，需根据实际数据调整
    
    返回:
        np.ndarray: 姿态角数组，形状为 (N, 3)，列顺序为 [roll, pitch, yaw]（单位：弧度）
    """
    # 确保输入是numpy数组
    quat_data = np.array(quat_data)
    
    # 调整四元数顺序为 [w, x, y, z]（标准四元数表示）
    # 注意：如果输入顺序是 [rot_x, rot_y, rot_z, rot_w]，则需重新排序
    w = quat_data[:, 3]  # rot_w
    x = quat_data[:, 0]  # rot_x
    y = quat_data[:, 1]  # rot_y
    z = quat_data[:, 2]  # rot_z
    
    # 计算roll (绕x轴旋转)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x**2 + y**2)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    # 计算pitch (绕y轴旋转)
    sinp = 2 * (w * y - z * x)
    # 处理万向锁情况（sinp接近±1）
    pitch = np.where(np.abs(sinp) >= 1, 
                    np.sign(sinp) * np.pi / 2,  # 限制在±90度
                    np.arcsin(sinp))
    
    # 计算yaw (绕z轴旋转)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y**2 + z**2)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    # 返回结果，形状为 (N, 3)
    return np.column_stack([roll, pitch, yaw])

def compute_physics_features(df):
    """计算去除重力后的线性加速度和姿态角，并只返回姿态角特征（roll, pitch, yaw）"""
    # 初始化存储姿态角的DataFrame
    attitude_angles_df = pd.DataFrame(index=df.index)  # 索引与原df一致

    # 按sequence_id分组计算姿态角
    for _, group in df.groupby('sequence_id'):
        
        # 计算姿态角
        quat_data = group[['rot_x', 'rot_y', 'rot_z', 'rot_w']].values
        attitude_angles = calculate_attitude_angles(quat_data)  # 返回 (N, 3) 数组
        
        # 将姿态角存入DataFrame（注意索引对齐）
        attitude_angles_df.loc[group.index, 'roll'] = attitude_angles[:, 0]    # 弧度
        attitude_angles_df.loc[group.index, 'pitch'] = attitude_angles[:, 1]   # 弧度
        attitude_angles_df.loc[group.index, 'yaw'] = attitude_angles[:, 2]     # 弧度

    # 返回仅包含姿态角的DataFrame（不计算linear_acc）
    return attitude_angles_df

def bin_tof_features_for_bfrb(df):
    """
    基于BFRB行为特点的TOF特征分箱
    包含所需特征的计算和分箱
    """
    binned_features = pd.DataFrame(index=df.index)
    
    # 先计算所需的TOF特征
    print("计算TOF必需特征...")
    tof_required_features = calculate_tof_required_features(df)
    
    # 将计算的特征合并到当前DataFrame中（仅用于本次分箱计算）
    df_with_features = pd.concat([df, tof_required_features], axis=1)
    
    # 1. 距离检测分箱（BFRB涉及特定身体部位接触）
    for i in [1, 2, 3, 4, 5]:  # 所有5个传感器
        feature_name = f'tof_{i}_depth_mean'
        if feature_name in df_with_features.columns:
            depth_data = df_with_features[feature_name]
            # 根据文献：BFRB通常涉及皮肤接触(很近距离)
            # 0-50mm: 皮肤接触, 50-100mm: 手部接近, 100-200mm: 中等距离, 200mm+: 远距离
            bins = [0, 50, 100, 200, np.inf]
            labels = [0, 1, 2, 3]  # 0=皮肤接触, 3=远距离
            binned_features[f'tof_{i}_distance_level'] = pd.cut(depth_data,
                                                               bins=bins,
                                                               labels=labels,
                                                               right=False)

    # 2. 表面粗糙度分箱（皮肤vs其他表面）
    for i in [2, 4]:  # 选择关键传感器（可根据实际效果调整）
        feature_name = f'tof_{i}_surface_roughness'
        if feature_name in df_with_features.columns:
            roughness_data = df_with_features[feature_name]
            # 根据文献：皮肤表面有特定的纹理模式
            # 0-0.01: 极光滑表面, 0.01-0.05: 皮肤纹理, 0.05-0.1: 粗糙表面, 0.1+: 很粗糙
            bins = [0, 0.01, 0.05, 0.1, np.inf]
            labels = [0, 1, 2, 3]  # 1=皮肤纹理特征
            binned_features[f'tof_{i}_surface_level'] = pd.cut(roughness_data,
                                                              bins=bins,
                                                              labels=labels,
                                                              right=False)

    # 3. 运动模式分箱（BFRB有特定的重复运动模式）
    if 'tof_motion_max' in df_with_features.columns:
        motion_data = df_with_features['tof_motion_max']
        # 根据文献：BFRB行为通常有特定的重复性运动模式
        # 0-0.1: 几乎无运动, 0.1-0.5: 轻微运动, 0.5-1.0: 中等运动, 1.0+: 剧烈运动
        bins = [0, 0.1, 0.5, 1.0, np.inf]
        labels = [0, 1, 2, 3]  # 2-3=可能的BFRB运动强度
        binned_features['tof_motion_intensity'] = pd.cut(motion_data,
                                                        bins=bins,
                                                        labels=labels,
                                                        right=False)
    
    return binned_features

def calculate_tof_required_features(df):
    """
    计算TOF分箱所需的特征
    """
    required_features_list = []
    
    for _, group in df.groupby('sequence_id'):
        required_features = pd.DataFrame(index=group.index)
        
        # 1. 计算每个传感器的深度均值
        for i in range(1, 6):  # 5个传感器
            cols = [f'tof_{i}_v{j}' for j in range(64)]
            if all(col in group.columns for col in cols):
                depths = group[cols].values
                required_features[f'tof_{i}_depth_mean'] = depths.mean(axis=1)
        
        # 2. 计算表面粗糙度
        for i in range(1, 6):
            cols = [f'tof_{i}_v{j}' for j in range(64)]
            if all(col in group.columns for col in cols):
                depths = group[cols].values
                
                roughness_values = []
                for depth_map in depths:
                    depth_map = depth_map.reshape(8, 8)
                    # 计算梯度幅值作为粗糙度（Sobel梯度更合适，这里用numpy简化）
                    grad_x = np.gradient(depth_map, axis=1)
                    grad_y = np.gradient(depth_map, axis=0)
                    # 根据文献，皮肤表面有特定的微观纹理
                    roughness = np.sqrt(grad_x**2 + grad_y**2).mean()
                    roughness_values.append(roughness)
                
                required_features[f'tof_{i}_surface_roughness'] = roughness_values
        
        # 3. 计算最大运动强度（基于光流）
        all_motion_energy = []
        sensor_motion_features = {}
        
        for i in range(1, 6):
            cols = [f'tof_{i}_v{j}' for j in range(64)]
            if all(col in group.columns for col in cols):
                images = group[cols].values.reshape(-1, 8, 8)
                
                if len(images) > 1:
                    # 计算帧间差分（简化光流）
                    flow = np.diff(images, axis=0)
                    flow = np.pad(flow, ((0,1), (0,0), (0,0)), mode='constant')
                else:
                    flow = np.zeros_like(images)
                
                # 计算运动能量
                motion_energy = (flow**2).mean(axis=(1,2))
                sensor_motion_features[f'tof_{i}_motion_energy'] = motion_energy
                all_motion_energy.append(motion_energy)
        
        # 取所有传感器中的最大值作为整体运动强度
        if all_motion_energy:
            required_features['tof_motion_max'] = np.max(np.array(all_motion_energy), axis=0)
        
        required_features_list.append(required_features)
    
    return pd.concat(required_features_list)


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

# ================================
# Training Pipeline
# ================================
if TRAIN:
    print("▶ TRAIN MODE – loading dataset …")
    df = pd.read_csv(RAW_DIR / "train.csv")

    df_demo = (pd.read_csv(RAW_DIR / "train_demographics.csv"))[['subject', 'handedness']]
    df = df.merge(df_demo, on='subject', how='left')

    df = df[~df['subject'].isin(["SUBJ_045235", "SUBJ_019262"])].reset_index(drop=True)

    print("  Transform the left hand into right hand")
    df = df.groupby('sequence_id', group_keys=False).apply(augment_left_handed_sequence)


    # Label encoding
    le = LabelEncoder()
    df['gesture_int'] = le.fit_transform(df['gesture'])
    np.save(EXPORT_DIR / "gesture_classes.npy", le.classes_)

    # Feature list
    meta_cols = {'gesture', 'gesture_int', 'sequence_type', 'behavior', 'orientation', 
                 'row_id', 'subject', 'phase', 'sequence_id', 'sequence_counter', 'handedness'}
    feature_cols_meta = [c for c in df.columns if c not in meta_cols]
    print("bin_tof_features_for_bfrb")
    binned_tof_features = bin_tof_features_for_bfrb(df)
    df = pd.concat([df, binned_tof_features], axis=1)
    print("calculate_tof_optical_flow")
    df_tof_flow=calculate_tof_optical_flow(df)
    df = pd.concat([df, df_tof_flow], axis=1)
    print("calculate_tof_3d_features")  
    df_tof_3d = calculate_tof_3d_features(df)
    df = pd.concat([df, df_tof_3d], axis=1)
    del df_tof_flow,df_tof_3d
    
    print("compute_physics_features")
    attitude_angles = compute_physics_features(df)
    df = df.merge(attitude_angles, left_index=True, right_index=True, how='left')
    print("add_rotated_acceleration_to_df")
    df = compute_global_acceleration_final(df, chunk_size=50000)
    print("  Removing gravity and calculating linear acceleration features...")

    linear_accel_list = []
    for _, group in df.groupby('sequence_id'):
        acc_data_group = group[['acc_x', 'acc_y', 'acc_z']]
        rot_data_group = group[['rot_x', 'rot_y', 'rot_z', 'rot_w']]
        linear_accel_group = remove_gravity_from_acc(acc_data_group, rot_data_group)
        linear_accel_list.append(pd.DataFrame(linear_accel_group, columns=['linear_acc_x', 'linear_acc_y', 'linear_acc_z'], index=group.index))
    df_linear_accel = pd.concat(linear_accel_list)
    # df_linear_accel = pd.read_parquet("./ver40/right/df_linear_accel.parquet")
    df = pd.concat([df, df_linear_accel], axis=1)

    print("  Calculating angular velocity from quaternion derivatives...")
    angular_vel_list = []
    for _, group in df.groupby('sequence_id'):
        rot_data_group = group[['rot_x', 'rot_y', 'rot_z', 'rot_w']]
        angular_vel_group = calculate_angular_velocity_from_quat(rot_data_group)
        angular_vel_list.append(pd.DataFrame(angular_vel_group, columns=['angular_vel_x', 'angular_vel_y', 'angular_vel_z'], index=group.index))
    df_angular_vel = pd.concat(angular_vel_list)
    df = pd.concat([df, df_angular_vel], axis=1)
    df['angvel_mag'] = np.sqrt(df['angular_vel_x']**2 + df['angular_vel_y']**2 + df['angular_vel_z']**2)
    print("  Calculating angular distance between successive quaternions...")
    angular_distance_list = []
    for _, group in df.groupby('sequence_id'):
        rot_data_group = group[['rot_x', 'rot_y', 'rot_z', 'rot_w']]
        angular_dist_group = calculate_angular_distance(rot_data_group)
        angular_distance_list.append(pd.DataFrame(angular_dist_group, columns=['angular_distance'], index=group.index))

    df_angular_distance = pd.concat(angular_distance_list)
    df = pd.concat([df, df_angular_distance], axis=1)
    print("Applying IMU binning...")
    imu_binned = bin_imu_features_for_bfrb(df)
    df = pd.concat([df, imu_binned], axis=1)
    del imu_binned,df_angular_distance,df_angular_vel
    # 定义简化后的统计特征列名
    imu_stat_col_simplified = ["acc_mean", "acc_max", "acc_std", "acc_min", "rot_mean", "rot_max", "rot_std", "rot_min"]
    stat_features = {}
    acc_grouped = df.groupby('sequence_id')[['acc_x', 'acc_y', 'acc_z']]
    stat_features['acc_mean'] = acc_grouped.mean().mean(axis=1) 
    stat_features['acc_max'] = acc_grouped.max().max(axis=1)    
    stat_features['acc_std'] = acc_grouped.std().mean(axis=1)   
    stat_features['acc_min'] = acc_grouped.min().min(axis=1)   
    rot_grouped = df.groupby('sequence_id')[['rot_x', 'rot_y', 'rot_z', 'rot_w']]
    stat_features['rot_mean'] = rot_grouped.mean().mean(axis=1) 
    stat_features['rot_max'] = rot_grouped.max().max(axis=1)     
    stat_features['rot_std'] = rot_grouped.std().mean(axis=1)  
    stat_features['rot_min'] = rot_grouped.min().min(axis=1)   

    stat_features_df = pd.DataFrame(stat_features)
    stat_features_df = stat_features_df.reset_index()
    df = df.merge(stat_features_df, on='sequence_id', how='left')
    
    imu_cols_base = [c for c in feature_cols_meta if not (c.startswith('thm_') or c.startswith('tof_'))]
    imu_engineered_features = [
    'linear_acc_x', 'linear_acc_y', 'linear_acc_z',
    'angular_vel_x', 'angular_vel_y', 'angular_vel_z',
    'angular_distance','acc_global_x','acc_global_y','acc_global_z','angvel_mag'
    ]
    IMU_BINNED_FEATURES = [
        'accel_intensity',
        'rotation_intensity',
        'motion_direction'
    ]
    imu_cols = imu_cols_base + imu_engineered_features+imu_stat_col_simplified+['roll','pitch','yaw']+IMU_BINNED_FEATURES

    thm_cols_original = [c for c in df.columns if c.startswith('thm_')]
    tof_aggregated_cols_template = []
    stat_types = ['mean', 'std', 'min', 'max', 'range', 'iqr', 'skew', 'kurtosis', 'mad']
    
    for i in range(1, 6):
        tof_aggregated_cols_template.extend([f'tof_{i}_{stat}' for stat in stat_types])
    

    print("  Calculating TOF/THM aggregated features...")
    df_tof_aggregated = pd.read_parquet("./ver40/right/tof_agg_replace1.parquet")
    df = pd.concat([df, df_tof_aggregated], axis=1)


    # tof_cols = thm_cols_original + tof_aggregated_cols_template
    # feature_cols = imu_cols + thm_cols_original + tof_aggregated_cols_template
    # 新增的 THM 特征列（差分、传感器间差异、趋势）
    TOF_FLOW_FEATURES = [
        'tof_1_flow_mean',
        'tof_1_motion_energy',
        'tof_3_flow_mean', 
        'tof_3_motion_energy',
        'tof_5_flow_mean',
        'tof_5_motion_energy',
        'tof_flow_mean_all',
        'tof_flow_std_all',
        'tof_motion_max',
        'tof_motion_sum'
    ]
    TOF_3D_FEATURES = [
        'tof_2_depth_mean',
        'tof_2_depth_std',
        'tof_2_depth_range',
        'tof_2_surface_roughness',
        'tof_4_depth_mean',
        'tof_4_depth_std',
        'tof_4_depth_range',
        'tof_4_surface_roughness',
        'tof_depth_mean_diff',
        'tof_depth_std_sum'
    ]
    TOF_ADVANCED_FEATURES = TOF_FLOW_FEATURES + TOF_3D_FEATURES
    TOF_BINNED_FEATURES = [
        'tof_1_distance_level','tof_2_distance_level','tof_3_distance_level', 
        'tof_4_distance_level','tof_5_distance_level',
        'tof_2_surface_level','tof_4_surface_level','tof_motion_intensity'
    ]
    # tof_cols = tof_aggregated_cols_template + tof_grid_cols + tof_diff_cols +thm_cols
    tof_cols = tof_aggregated_cols_template+thm_cols_original+TOF_ADVANCED_FEATURES+TOF_BINNED_FEATURES
    feature_cols = imu_cols +tof_cols

    print(f"  IMU {len(imu_cols)} | TOF/THM {len(tof_cols)} | total {len(feature_cols)} features")

    
    # Build sequences
    # df= enhance_thm_tof_features(df)
    seq_gp = df.groupby('sequence_id')
    all_steps_for_scaler_list = []
    X_list, y_list, id_list, lens = [], [], [], []
    for seq_id, seq in tqdm(seq_gp):
        seq_df = seq.copy()
        for i in range(1, 6):
            pixel_cols_tof = [f"tof_{i}_v{p}" for p in range(64)]
            tof_sensor_data = seq_df[pixel_cols_tof].replace(-1, np.nan)
            seq_df[f'tof_{i}_mean'] = tof_sensor_data.mean(axis=1)
            seq_df[f'tof_{i}_std']  = tof_sensor_data.std(axis=1)
            seq_df[f'tof_{i}_min']  = tof_sensor_data.min(axis=1)
            seq_df[f'tof_{i}_max']  = tof_sensor_data.max(axis=1)
            seq_df[f'tof_{i}_range'] = tof_sensor_data.max(axis=1) - tof_sensor_data.min(axis=1)
            seq_df[f'tof_{i}_iqr'] = tof_sensor_data.quantile(0.75, axis=1) - tof_sensor_data.quantile(0.25, axis=1)
            seq_df[f'tof_{i}_skew'] = tof_sensor_data.skew(axis=1)
            seq_df[f'tof_{i}_kurtosis'] = tof_sensor_data.kurtosis(axis=1)
            seq_df[f'tof_{i}_mad'] = tof_sensor_data.sub(tof_sensor_data.mean(axis=1), axis=0).abs().mean(axis=1)

        mat = seq_df[feature_cols].ffill().bfill().fillna(0).values.astype('float32')
        all_steps_for_scaler_list.append(mat)
        X_list.append(mat)
        y_list.append(seq['gesture_int'].iloc[0])
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
    X_list_all = pad_sequences_torch(X_list, maxlen=pad_len, padding='pre', truncating='pre')
    y_list_all = np.eye(len(le.classes_))[y_list].astype(np.float32)  # One-hot encoding


    augmenter = Augment(
        p_jitter=0.9844818619033621, sigma=0.03291295776089293, scale_range=(0.7542342630597011,1.1625052821731077),
        p_dropout=0.41782786013520684,
        p_moda=0.3910622476959722, drift_std=0.0040285239353308015, drift_max=0.3929358950258158    
    )


metric_loss_fn = losses.TripletMarginLoss(margin=0.2) 
miner = miners.TripletMarginMiner(margin=0.2, type_of_triplets="hard")

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
    model = FuseGate(maxlen, len(imu_cols), len(tof_cols), 
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
    print(f"▶ Starting Fold {fold} training...")
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

        train_acc = CompetitionMetric().calculate_hierarchical_f1(
            pd.DataFrame({'gesture': le.classes_[train_targets]}),
            pd.DataFrame({'gesture': le.classes_[train_preds]}))
        val_acc = CompetitionMetric().calculate_hierarchical_f1(
            pd.DataFrame({'gesture': le.classes_[val_targets]}),
            pd.DataFrame({'gesture': le.classes_[val_preds]}))
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
    
    eval_model = FuseGate(maxlen, len(imu_cols), len(tof_cols), 
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

    h_f1 = CompetitionMetric().calculate_hierarchical_f1(
        pd.DataFrame({'gesture': le.classes_[val_targets]}),
        pd.DataFrame({'gesture': le.classes_[val_preds]})
    )

    all_h_f1 = CompetitionMetric().calculate_hierarchical_f1(
        pd.DataFrame({'gesture': le.classes_[val_targets]}),
        pd.DataFrame({'gesture': le.classes_[val_preds2]})
    )

    print(f"Fold {fold+1} H-F1 = {h_f1:.4f}")
    print(f"Fold {fold+1} All H-F1 = {all_h_f1:.4f}")
    print("\n")


imu_h_f1 = CompetitionMetric().calculate_hierarchical_f1(
    pd.DataFrame({'gesture': le.classes_[oof_imu.astype(int)]}),
    pd.DataFrame({'gesture': le.classes_[y_list_all.argmax(axis=1)]})
)

h_f1 = CompetitionMetric().calculate_hierarchical_f1(
    pd.DataFrame({'gesture': le.classes_[oof.astype(int)]}),
    pd.DataFrame({'gesture': le.classes_[y_list_all.argmax(axis=1)]})
)

print(f"OOF H-F1 = {h_f1:.4f}")
print(f"OOF IMU H-F1 = {imu_h_f1:.4f}")

# Fold 5 ori

# OOF H-F1 = 0.8758
# OOF IMU H-F1 = 0.8491

# 0.8624691571465476

# Right Handed OOF H-F1 = 0.8802
# Right Handed OOF IMU H-F1 = 0.8534
# Right Handed OOF H-F1 + IMU H-F1 = 0.8668


# Left Handed OOF H-F1 = 0.8463
# Left Handed OOF IMU H-F1 = 0.8197
# Left Handed OOF H-F1 + IMU H-F1 = 0.8330

(imu_h_f1 + h_f1) / 2

trian_dem = pd.read_csv(RAW_DIR / "train_demographics.csv")
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


groups_by_subject = seq_gp['subject'].first()

skf = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=42)

oof = np.zeros((X_list_all.shape[0], 18), dtype=np.float32)
oof_imu = np.zeros((X_list_all.shape[0], 18))
for fold, (train_idx, val_idx) in enumerate(skf.split(id_list, seq_gp['sequence_type'].first(), groups=groups_by_subject)):
    _, val_list = X_list_all[train_idx], X_list_all[val_idx]
    _, val_y_list = y_list_all[train_idx], y_list_all[val_idx]

    val_dataset = CMI3Dataset(val_list, val_y_list, maxlen, mode="val")
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,drop_last=False)
    

    # best_model_path = Path("/kaggle/input/cmi-clear-imu-only") / f"exp172_gesture_two_branch_fold{fold}.pth"
    best_model_path = EXPORT_DIR/ f"gesture_two_branch_fold{fold}.pth"
    print(f"Evaluating fold {fold+1} with model {best_model_path}")
    
    eval_model = FuseGate(maxlen, len(imu_cols), len(tof_cols), 
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
            val_preds2.extend(logits2.cpu().numpy())
        
            half = BATCH_SIZE // 2         

            x_front = X[:half].clone()                 
            x_back  = X[half:].clone()      
            
            x_front[:, :, len(imu_cols):] = 0.0  
            x_back[:, :, len(imu_cols):] = 0.0    
            X = torch.cat([x_front, x_back], dim=0)  # (B, C, T)
            X, y = X.float().to(device), y.to(device)
            
            logits = eval_model(X)[0]
            val_preds.extend(logits.cpu().numpy())
            val_targets.extend(y.argmax(dim=1).cpu().numpy())

    oof_imu[val_idx] = val_preds
    oof[val_idx] = val_preds2



    h_f1 = CompetitionMetric().calculate_hierarchical_f1(
        pd.DataFrame({'gesture': le.classes_[val_targets]}),
        pd.DataFrame({'gesture': le.classes_[np.stack(val_preds, axis=0).argmax(axis=1)]})
    )

    all_h_f1 = CompetitionMetric().calculate_hierarchical_f1(
        pd.DataFrame({'gesture': le.classes_[val_targets]}),
        pd.DataFrame({'gesture': le.classes_[np.stack(val_preds2, axis=0).argmax(axis=1)]})
    )

    print(f"Fold {fold+1} H-F1 = {h_f1:.4f}")
    print(f"Fold {fold+1} All H-F1 = {all_h_f1:.4f}")
    print("\n")


imu_h_f1 = CompetitionMetric().calculate_hierarchical_f1(
    pd.DataFrame({'gesture': le.classes_[oof_imu.argmax(axis=1).astype(int)]}),
    pd.DataFrame({'gesture': le.classes_[y_list_all.argmax(axis=1)]})
)

h_f1 = CompetitionMetric().calculate_hierarchical_f1(
    pd.DataFrame({'gesture': le.classes_[oof.argmax(axis=1).astype(int)]}),
    pd.DataFrame({'gesture': le.classes_[y_list_all.argmax(axis=1)]})
)

print(f"OOF H-F1 = {h_f1:.4f}")
print(f"OOF IMU H-F1 = {imu_h_f1:.4f}")

# %%
pd.DataFrame(oof_imu, columns=['logit_' + col for col in list(le.classes_)]).to_csv(EXPORT_DIR / "oof_ver40_imu.csv", index=False)


