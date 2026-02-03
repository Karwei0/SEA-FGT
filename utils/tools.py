import os
import smtplib
import time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import matplotlib.pyplot as plt
import seaborn as sns


import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

plt.switch_backend('agg')


# convert bytes into readable format
def convert_size(size_bytes):
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024:
            return f'{size_bytes:.2f} {unit}'
        size_bytes /= 1024.0
    return f'{size_bytes:.2f} PB'

# remove txt files
def delete_txt_files_in_folder(path):
    [os.remove(os.path.join(path, f)) for f in os.listdir(path) if f.endswith('.txt')]

def dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

# mat to excel
def write_into_xls(excel_name, mat, columns=None):
    file_extension = os.path.splitext(excel_name)[1]

    if file_extension != '.xls' and file_extension != '.xlsx':
        raise ValueError

    folder_name = os.path.dirname(excel_name)
    if folder_name:
        os.makedirs(folder_name, exist_ok=True)

    if isinstance(mat, np.ndarray) and mat.ndim > 2:
        mat = mat.reshape(-1, mat.shape[-1])
        mat = mat[:1000]
    if columns is not None:
        dataframe = pd.DataFrame(mat, columns=columns)
    else:
        dataframe = pd.DataFrame(mat)

    dataframe.to_excel(excel_name, index=False)

# plot
def visual(true, preds=None, name='./pic/test.pdf', imp=False):
    folder_name = os.path.dirname(name)
    if folder_name:
        os.makedirs(folder_name, exist_ok=True)
    label2 = 'Imputation' if imp else 'Prediction'

    # look out
    if not isinstance(true, np.ndarray):
        true = true.numpy()
    if not isinstance(preds, np.ndarray):
        preds = preds.numpy()

    plt.figure()
    plt.plot(true, label='Ground Truth', linestyle='--', linewidth=2)
    if preds is not None:
        plt.plot(preds, label=label2, linewidth=2)
    plt.legend()
    plt.grid(linestyle=':', color='lightgray')
    plt.savefig(name, bbox_inches='tight')

def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)

# find the most recently modified subfolder -- maybe a well-trained model
def find_most_recently_modified_subfolder(base_dir, file_name='checkpoint.pth', contain_str=''):
    most_recent_time = 0
    most_recent_folder = None
    most_recent_subfolder = None

    if isinstance(contain_str, list):
        subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and
                   os.path.isfile(os.path.join(base_dir, d, file_name)) and all([cstr in d for cstr in contain_str])]
    else:
        subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and
                   os.path.isfile(os.path.join(base_dir, d, file_name)) and contain_str in d]

    for subdir in subdirs:
        folder_path = os.path.join(base_dir, subdir)
        current_time = os.path.getmtime(folder_path)

        if current_time > most_recent_time:
            most_recent_time = current_time
            most_recent_folder = folder_path
            most_recent_subfolder = subdir
    return most_recent_folder, most_recent_subfolder

def compare_prefix_before_third_underscore(str1, str2, num=3):
    if str1 is None or str2 is None:
        return False
    prefix1 = ''.join(str1.split('-', num)[:num])
    prefix2 = ''.join(str2.split('-', num)[:num])

    are_prefixes_equal = prefix1 == prefix2
    return are_prefixes_equal

def compute_gradient_norm(model):
    total_norm = 0.0
    for param in model.parameters():
        if param.requires_grad and param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm

# check if a list is empty or contains NaN values
def is_not_empty_or_nan(a):
    if isinstance(a, list):
        if not a:
            return False
        if any(isinstance(i, (float, np.float32, np.float64)) and np.isnan(i) for i in a):
            return False
    elif isinstance(a, torch.Tensor):
        if a.numel() == 0:
            return False
        if torch.isnan(a).any():
            return False
    else:
        if isinstance(a, (float, np.float32, np.float64)) and np.isnan(a):
            return False
    return True

def cosine_distance(tensor1, tensor2, keepdims=False):
    assert tensor1.shape == tensor2.shape, 'two tensors should have the same shape'
    cosine_sim = F.cosine_similarity(tensor1, tensor2, dim=-1, eps=1e-8)
    cosine_dist = 1 - cosine_sim
    if keepdims:
        return cosine_dist.unsqueeze(-1)
    else:
        return cosine_dist

def euclidean_distance(tensor1, tensor2, keepdims=False):
    assert tensor1.shape == tensor2.shape, 'two tensors should have the same shape'
    diff = tensor1 - tensor2
    squared_diff = torch.square(diff)
    euclidean_dist = torch.sqrt(torch.sum(squared_diff, dim=-1))
    if keepdims:
        return euclidean_dist.unsqueeze(-1)
    else:
        return euclidean_dist

def send_email(subject='Python Notification', body='Program Complete!!!!', to_email=r'xxx',
               from_email=r'xxxx@xxx', password='xxxxx', mail_host='xxx', mail_port=465):
    message = MIMEMultipart()
    message['From'] = from_email
    message['To'] = to_email
    message['Subject'] = subject
    message.attach(MIMEText(body, 'plain', 'utf-8'))


    try:
        # Connect to the SMTP server using SSL(port 465)
        with smtplib.SMTP_SSL(mail_host, mail_port) as server:
            server.login(from_email, password)
            server.send_message(message)
        print("Email sent successfully.")
    except Exception as e:
        print("Error sending email: ", e)

def create_sub_diagonal_matrix(n, value=1, offset=0):
    if abs(offset) >= n:
        return None
    vec = torch.ones(n - abs(offset)) * value
    return torch.diag(vec, diagonal=offset)

def plot_mat(mat, str_cat='series_2D', str0='tmp', save_folder='./results'):
    if not isinstance(mat, np.ndarray):
        mat = mat.detach().cpu().numpy()
    if not os.path.exists(save_folder):
        os.makedirs(save_folder, exist_ok=True)

    plt.figure(figsize=(8, 8))
    sns.heatmap(mat, annot=False, cmap='coolwarm', square=True, cbar=True)
    plt.xticks([]) # remove ticks
    plt.yticks([])

    timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    plt.savefig(os.path.join(save_folder, f'{str_cat}_{str0}-{timestamp}.pdf'))
    plt.show()

    excel_name = os.path.join(save_folder, f'{str_cat}_{str0}-{timestamp}.xlsx')
    write_into_xls(excel_name, mat)
    np.save(os.path.join(save_folder, f'{str_cat}_{str0}-{timestamp}.npy'), mat)

if __name__ == '__main__':
    # === 填你自己的邮箱信息 ===
    FROM = "your_email@example.com"
    TO = "receiver@example.com"
    PWD = "your_authorization_code"  # 授权码，不是登录密码
    HOST = "smtp.example.com"
    PORT = 465

    send_email(
        subject="测试邮件：send_email()",
        body="你好，这是来自 tools.py 的自动测试邮件。\n如果你收到这封信，说明函数可用。",
        to_email=TO,
        from_email=FROM,
        password=PWD,
        mail_host=HOST,
        mail_port=PORT
    )
