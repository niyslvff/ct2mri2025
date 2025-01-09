import sys
import os
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append("..")

import torch
from torch.utils.data import DataLoader
import gaussian_diffusion
from tqdm import tqdm
import numpy as np
from models.vqvae.vq_vae import VQVAE
import argparse
from typing import TypeVar, Optional
import datetime
from utils import showSlices
Tensor = TypeVar("torch.tensor")

class Test:
    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        self.in_channels = kwargs.get("in_channels")
        self.inner_channel = kwargs.get('inner_channel')
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.dataset_name = kwargs.get("dataset_name")
        self.image_size = kwargs.get("image_size")
        self.image_root = kwargs.get("image_root")
        self.use_vae = kwargs.get("use_vae")
        self.dims = kwargs.get("dims")
        self.sample_method = kwargs.get("sample_method")
        self.model_name = kwargs.get("model_name")
        self.version = kwargs.get("version")
        self.organ = kwargs.get("organ")
        self.vae_weight_file = kwargs.get("vae_weight_file")
        self.vae_embedding_dims = kwargs.get("vae_embedding_dims")
        self.generate_concatenated_volume = kwargs.get("generate_concatenated_volume")
        self.double_channels = kwargs.get("double_channels")
        
        self.dataset = self.choose_dataset()
        self.dataloader = DataLoader(self.dataset, 1)
        
        # 加载扩散模型
        self.model = self.choose_model()
        self.model.to(self.device)
        checkpoints_file = torch.load(f"../train/{self.model_name}/weight_file/{self.organ}/{self.version}/best.pth", map_location=self.device)
        try:
            self.model.load_state_dict(checkpoints_file["model_parameters"])
        except:
            self.model.load_state_dict(checkpoints_file["model"])
        print(f"model best loss: {checkpoints_file['best_epoch']}, best loss: {checkpoints_file['best_loss']}")
        self.model.eval()
        
        # 加载vae模型
        self.vae_model = VQVAE(
            in_channels=1,
            out_channels=1,
            embedding_dim=self.vae_embedding_dims,
            num_embeddings=512,
            beta=0.25,
            dims=3,
        )
        self.vae_model.to(self.device)
        vae_checkpoints_file = torch.load(self.vae_weight_file)
        try:
            self.vae_model.load_state_dict(vae_checkpoints_file['model'])
        except:
            self.vae_model.load_state_dict(vae_checkpoints_file['model_parameters'])
            
        print(f"vae best epoch: {vae_checkpoints_file['best_epoch']}, best loss: {vae_checkpoints_file['best_loss']}")
        self.vae_model.eval()
        
        self.timestep_amount = kwargs.get("timestep_amount")
        self.gaussian_scheduler = gaussian_diffusion.DDPMScheduler(
            num_train_timesteps=self.timestep_amount
        )
    
    def choose_model(self):
        if self.model_name.lower() == "sunet":
            if self.version == "v2":
                from models.sunet.sunet_v2 import UNet
            elif "v1_1" in self.version:
                from models.sunet.sunet_v1_1 import UNet
            elif "v1_2" in self.version:
                from models.sunet.sunet_v1_2 import UNet
            elif "v1" in self.version:
                from models.sunet.sunet_v1 import UNet
            else:
                raise ValueError("wrong sunet version.")
            return UNet(
                in_channel=self.in_channels * 2 if self.double_channels else self.in_channels,
                out_channel=2 if self.double_channels else 1,
                inner_channel=self.inner_channel,
                channel_mults=(1, 2, 4, 8, 8),
                attn_res=(3, 4),
                dims=self.dims
            )
        elif self.model_name.lower() == "dit":
            from models.dit.models import DiT
            return DiT(
                input_size=(168, 224),
                patch_size=8,
                hidden_channels=256,
                in_channel=self.in_channels,
                out_channels=1,
            )
        else:
            raise ValueError("wrong model.")
    
    def choose_dataset(self):
        from dataset import MedicalDataset3D, Raw3dDiffusionConcatenateDataset, Raw3dDiffusionMajiDataset
        if self.dataset_name == "MedicalDataset3D":
            return MedicalDataset3D(
                validate=True,
            )
        elif self.dataset_name == "concatenation":
            return Raw3dDiffusionConcatenateDataset(
                img_num=100,
                img_root_path=self.image_root,
                is_val_set=True,
                img_shape=self.image_size,
                use_crop=False,
                transform=None,
            )
        elif self.dataset_name == "Raw3dDiffusionMajiDataset":
            return Raw3dDiffusionMajiDataset(
                img_num=1,
                img_shape=self.image_size,
                img_root_path=self.image_root,
                use_crop=False,
                is_val=True,
                is_used_to_concate=self.generate_concatenated_volume,
            )
        else:
            raise ValueError("wrong dataset.")
        
    @torch.no_grad()
    def test_one(self, batch: tuple[Tensor, Tensor], save_path: str, if_loop: bool = False, idx: int = 0):
        ct, mr = batch
        ct_image, mr_image = ct.squeeze().squeeze(), mr.squeeze().squeeze()
        ct, mr = ct.to(self.device), mr.to(self.device)
        # 完全随机生成
        # mr_fake = torch.randn((1, 1, *self.image_size), device=self.device, dtype=torch.float32)
        # 条件随机生成
        ct_fake = torch.clone(mr)
        
        mrs, ct_fakes = [], []
        if self.use_vae:
            mr = self.vae_model.encode(mr)[0]
            ct_fake = self.vae_model.encode(ct_fake)[0]
            if not self.double_channels:
                for i in range(ct.shape[1]):
                    mrs.append(ct[:, i, :, :, :].unsqueeze(1))
                    ct_fakes.append(ct_fake[:, i, :, :, :].unsqueeze(1))
            else:
                mrs.append(mr)
                ct_fakes.append(ct_fake)
             
        pred_ct_fakes = []     
        for i_channels in range(len(mrs)):
            with tqdm(total=self.timestep_amount) as tq:  
                tq.set_description("{:d}/{:d}/{:d}".format(i_channels, idx, self.dataset.__len__()))
                ct_fake = ct_fakes[i_channels]
                mr = mrs[i_channels]
                print(ct_fake.shape, mr.shape)
                if self.sample_method == "ddpm":
                    for i in range(0, self.timestep_amount)[::-1]:
                        t = torch.full((1,), i, device=self.device, dtype=torch.long)
                        pred_noise = self.model(ct_fake, mr, t) if self.in_channels == 2 else self.model(ct_fake, t)
                        ct_fake = self.gaussian_scheduler.resample(pred_noise, ct_fake, t)
                        ct_fake = torch.clamp(ct_fake, -1.0, 1.0)
                        tq.set_postfix(t=t)
                        tq.update(1)
                else:
                    ddim_step = list(range(200, self.timestep_amount)[::-200])
                    # ddim_step = [999, 500, 200]
                    ddim_step += list(range(0, 201)[::-10])
                    n = 0
                    for i in ddim_step:
                        t = torch.full((1,), i, device=self.device, dtype=torch.long)
                        t_prev = torch.full((1,), i - int(ddim_step[n] - ddim_step[n + 1]) if i >= ddim_step[-2] else ddim_step[-1], device=self.device, dtype=torch.long)
                        noise_pred = self.model(ct_fake, mr, t) if self.in_channels == 2 else self.model(ct_fake, t)
                        ct_fake = self.gaussian_scheduler.denoise_ddim(noise_pred, ct_fake, t, t_prev)
                        ct_fake = torch.clamp(ct_fake, -1.0, 1.0)
                        if n < len(ddim_step) - 1:
                            tq.update(ddim_step[n] - ddim_step[n + 1])
                        else:
                            tq.update(ddim_step[-1])
                        n += 1
            pred_ct_fakes.append(ct_fake)
        
        ct_fake = torch.cat(tuple(pred_ct_fakes), dim=1)
        if self.use_vae:
            quantized_fake, _ = self.vae_model.vq_layer(ct_fake)
            ct_fake = self.vae_model.decode(quantized_fake)
            
        ct_fake_image = np.squeeze(np.squeeze(ct_fake.cpu(), axis=0), axis=0)
        return ct_image, ct_fake_image

    
    def test_loop(self, if_loop: bool = True, name: Optional[str] = None):
        # 创建预测结果文件夹
        # os.makedirs("prediction/real", exist_ok=True)
        if not name:
            current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"prediction/{self.organ}/{self.model_name}.{self.version}.{self.sample_method}.{current_time}/fake"
        os.makedirs(save_path, exist_ok=True)
        
        for idx, batch in enumerate(self.dataloader):
            # batch[0]是条件图片，batch[1]是目标图片。
            if self.generate_concatenated_volume:
                result_3d_z = torch.zeros(self.image_size)
                for i in range(self.image_size[0]):
                    new_batch = (batch[0][:, :, i, :, :], batch[1][:, :, i, :, :])
                    original_ct_image, fake_ct_image = self.test_one(
                        batch=new_batch, 
                        save_path=save_path,
                        if_loop=if_loop,
                        idx=idx,
                    )
                    result_3d_z[i, :, :] = fake_ct_image
                result_3d_z.detach().numpy().tofile(f"{save_path}/{idx}.raw")
                fake_ct_volume = result_3d_z
                original_ct_volume = batch[1].squeeze().squeeze()
            else:   
                if self.dims == 2 and len(batch[0].shape) == 5:
                    batch = (batch[0][:, :, self.image_size[0] // 2, :, :], batch[1][:, :, self.image_size[0] // 2, :, :])
                original_ct_volume, fake_ct_volume = self.test_one(
                    batch=batch, 
                    save_path=save_path,
                    if_loop=if_loop,
                    idx=idx,
                )
                fake_ct_volume.detach().numpy().tofile(f"{save_path}/{idx}.raw")
                
            if not if_loop:
                slices = []
                for volume in [original_ct_volume, fake_ct_volume]:
                    if self.dims == 3 or self.generate_concatenated_volume:
                        slices.append(volume[volume.shape[0] // 2, :, :])
                        slices.append(volume[:, :, volume.shape[2] // 2])
                        slices.append(volume[:, volume.shape[1] // 2, :])
                    else:
                        slices.append(volume)
                showSlices(slices, 2 if self.dims == 3 or self.generate_concatenated_volume else 1, f"{os.path.dirname(save_path)}/test.png")
                return
            
            if idx == 29:
                return
            


class BaseCommand:
    r"""
    头颅:
    测试sunet_v1_2: python3 test_diff.py --version=v1_2_3d --dataset_name=concatenation --use_vae=1 --image_root=../crop/concatenate_mri_pairs_brain
    测试sunet_v2: python3 test_diff.py --version=v2 --in_channels=1
    测试vqvae_v1(随机剪裁训练8000epochs)+sunet_v1(使用twoloss训练): 
    python3 test_diff.py --version=v1_twoloss_3d --dataset=concatenation --use_vae=1 --dims=3 --inner_channel=64 --image_root=../crop/concatenate_mri_pairs_brain --vae_weight_file=../train/vqvae/weight_file/brain/v1/best.pth
    单张测试ddim:
    python3 test_diff_mri2ct.py --version=sunet.v1.3d.latent.20241028 --dataset=Raw3dDiffusionMajiDataset --use_vae=1 --inner_channel=64 --dims=3 --double_channels=1 --if_loop=1 --vae_embedding_dims=2 --sample_method=ddim --image_root=../crop/preprocess_globalNormAndEnContrast --vae_weight_file=../train/vqvae/weight_file/brain/v1/best.pth


    骨盆:
    测试sunet_v1_2d, 循环生成层叠体，使用ddim采样:
    python3 test_diff.py --organ=pelvis --generate_concatenated_volume=1 --depth=96 --height=240 --width=384 --dims=2 --if_loop=0 --version=sunet.v1.2d.20241003 --dataset_name=Raw3dDiffusion160224168Dataset --image_root=../crop/preprocess_globalNormAndEnContrast_pelvis --sample_method=ddim
    测试sunet_v1_twoloss_3d，使用ddim:
    python3 test_diff.py --organ=pelvis --dims=3 --version=sunet.v1.3d.latent.20241004 --dataset_name=concatenation --use_vae=1 --inner_channel=64 --if_loop=1 --sample_method=ddim --depth=96 --height=240 --width=384 --image_root=../crop/concatenate_mri_pairs_pelvis --vae_weight_file=../train/vqvae/weight_file/pelvis/v1/best.pth


    """
    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser()
        self.add_base_arguments()
        
    def add_base_arguments(self):
        # 添加基础参数
        self.parser.add_argument('--timestep_amount', type=int, default=1000)
        self.parser.add_argument('--organ', type=str, choices=["brain", "pelvis"], default="brain")
        self.parser.add_argument('--dataset_name', type=str, default="MedicalDataset3D")
        self.parser.add_argument('--image_root', type=str, default="../crop/preprocess_globalNormAndEnContrast")
        self.parser.add_argument('--depth', type=int, default=160)
        self.parser.add_argument('--height', type=int, default=224)
        self.parser.add_argument('--width', type=int, default=168)
        self.parser.add_argument('--model_name', type=str, default="sunet")
        self.parser.add_argument('--version', type=str, default="v1")
        self.parser.add_argument('--in_channels', type=int, default=2)
        self.parser.add_argument('--double_channels', type=int, choices=[0, 1], default=1)
        self.parser.add_argument('--inner_channel', type=int, default=32)
        self.parser.add_argument('--use_vae', type=int, choices=[0, 1], default=0)
        self.parser.add_argument('--vae_weight_file', type=str, default="../VAE/weight_file/vqvae_3d_v2/best.pth")
        self.parser.add_argument('--vae_embedding_dims', type=int, default=2)
        self.parser.add_argument('--dims', type=int, choices=[2, 3], default=3)
        self.parser.add_argument('--sample_method', type=str, choices=["ddpm", "ddim"], default="ddpm")
        self.parser.add_argument('--if_loop', type=int, choices=[0, 1], default=0)
        self.parser.add_argument('--generate_concatenated_volume', type=int, choices=[0, 1], default=0)
        
    def parse_args(self):
        args = self.parser.parse_args()
        args.use_vae = bool(int(args.use_vae))
        args.if_loop = bool(int(args.if_loop))
        args.double_channels = bool(int(args.double_channels))
        args.generate_concatenated_volume = bool(int(args.generate_concatenated_volume))
        args.image_size = (args.depth, args.height, args.width)
        return args
        


if __name__ == "__main__":
    command_parse = BaseCommand()
    args = command_parse.parse_args()
    print(args)
    
    print("是否开始测试？(1/0)")
    if int(input()) == 1:
        t = Test(**vars(args))
        t.test_loop(if_loop=args.if_loop)
