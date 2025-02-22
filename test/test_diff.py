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
from torchvision.transforms import transforms
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
        self.use_collate = kwargs.get("use_collate")
        self.if_best = kwargs.get("if_best")
        
        self.dataset = self.choose_dataset()
        self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
        ])
        self.dataloader = DataLoader(self.dataset, 1, collate_fn=self.dataloader_collate_fn_2d if self.use_collate else None,)
        
        # 加载扩散模型
        self.model = self.choose_model()
        self.model.to(self.device)
        checkpoints_file = torch.load(f"../train/{self.model_name}/weight_file/{self.organ}/{self.version}/{'best.pth' if self.if_best else 'latest.pth'}", map_location=self.device)
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
        elif self.model_name.lower() == "vit":
            from models.vit.models import ViT
            return ViT(
                input_size=(168, 224),
                patch_size=8,
                hidden_channels=256,
                in_channel=self.in_channels,
                out_channels=1,
            )
        elif self.model_name.lower() == "dit":
            from models.dit.models import DiT_models
            return DiT_models['DiT-S/8'](
                input_size=(224, 224), 
                in_channels=1,
                learn_sigma=False,
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

    def dataloader_collate_fn_2d(self, batch: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        r"""输入batch前重新对数据对进行操作，
        从每个体数据中抽取一个切片组成新的batch，输出形状都相同。
        """
        
        N = len(batch)
        C, W, D, H = batch[0][0].shape
        random_slices = torch.randint(0, D, (N,), dtype=torch.int)
        original_ct_tensors = torch.stack(
            [
                tensor[0][:, :, slice_num, :]
                for tensor, slice_num in zip(batch, random_slices)
            ]
        )
        original_mr_tensors = torch.stack(
            [
                tensor[1][:, :, slice_num, :]
                for tensor, slice_num in zip(batch, random_slices)
            ]
        )
        
        seed = torch.random.seed()
        torch.random.manual_seed(seed)
        transformed_ct_tensors = torch.stack(
            [
                self.transform(tensor[0][:, :, slice_num, :])
                for tensor, slice_num in zip(batch, random_slices)
            ]
        )
        torch.random.manual_seed(seed)
        transformed_mr_tensors = torch.stack(
            [
                self.transform(tensor[1][:, :, slice_num, :])
                for tensor, slice_num in zip(batch, random_slices)
            ]
        )

        return transformed_ct_tensors, transformed_mr_tensors
        
    @torch.no_grad()
    def test_one(self, batch: tuple[Tensor, Tensor], save_path: str, if_loop: bool = False, idx: int = 0):
        ct, mr = batch
        ct_image, mr_image = ct.squeeze().squeeze(), mr.squeeze().squeeze()
        ct, mr = ct.to(self.device), mr.to(self.device)
        # 完全随机生成
        # mr_fake = torch.randn((1, 1, *self.image_size), device=self.device, dtype=torch.float32)
        # 条件随机生成
        mr_fake = torch.clone(ct)
        
        cts, mr_fakes = [], []
        if self.use_vae:
            ct = self.vae_model.encode(ct)[0]
            mr_fake = self.vae_model.encode(mr_fake)[0]
            if not self.double_channels:
                for i in range(ct.shape[1]):
                    cts.append(ct[:, i, :, :, :].unsqueeze(1))
                    mr_fakes.append(mr_fake[:, i, :, :, :].unsqueeze(1))
            else:
                cts.append(ct)
                mr_fakes.append(mr_fake)
        else:
            cts.append(ct)
            mr_fakes.append(mr_fake)
             
        pred_mr_fakes = []     
        for i_channels in range(len(cts)):
            with tqdm(total=self.timestep_amount) as tq:  
                tq.set_description("{:d}/{:d}/{:d}".format(i_channels, idx, self.dataset.__len__()))
                mr_fake = mr_fakes[i_channels]
                ct = cts[i_channels]
                if self.sample_method == "ddpm":
                    if self.model_name == "dit":
                        # ddpm_step = [999, 799, 599, 399, 199] + list(range(0, 200)[::-1])
                        ddpm_step = list(range(0, self.timestep_amount)[::-1])
                    else:
                        ddpm_step = list(range(0, self.timestep_amount)[::-1])
                    for i in ddpm_step:
                        t = torch.full((1,), i, device=self.device, dtype=torch.long)
                        pred_noise = self.model(mr_fake, ct, t) if self.in_channels == 2 else self.model(mr_fake, t)
                        mr_fake = self.gaussian_scheduler.resample(pred_noise, mr_fake, t)
                        mr_fake = torch.clamp(mr_fake, -1.0, 1.0)
                        tq.set_postfix(t=t)
                        tq.update(1)
                else:
                    if self.model_name == "dit":
                        ddim_step = list(range(200, self.timestep_amount)[::-400])
                        ddim_step += list(range(0, self.timestep_amount)[::-20])
                    else:  
                        ddim_step = list(range(200, self.timestep_amount)[::-200])
                        ddim_step += list(range(0, 201)[::-10])
                    n = 0
                    for i in ddim_step:
                        t = torch.full((1,), i, device=self.device, dtype=torch.long)
                        t_prev = torch.full((1,), i - int(ddim_step[n] - ddim_step[n + 1]) if i >= ddim_step[-2] else ddim_step[-1], device=self.device, dtype=torch.long)
                        noise_pred = self.model(mr_fake, ct, t) if self.in_channels == 2 else self.model(mr_fake, t)
                        mr_fake = self.gaussian_scheduler.denoise_ddim(noise_pred, mr_fake, t, t_prev)
                        mr_fake = torch.clamp(mr_fake, -1.0, 1.0)
                        if n < len(ddim_step) - 1:
                            tq.update(ddim_step[n] - ddim_step[n + 1])
                        else:
                            tq.update(ddim_step[-1])
                        n += 1
            pred_mr_fakes.append(mr_fake)
        
        mr_fake = torch.cat(tuple(pred_mr_fakes), dim=1)
        if self.use_vae:
            quantized_fake, _ = self.vae_model.vq_layer(mr_fake)
            mr_fake = self.vae_model.decode(quantized_fake)
            
        mr_fake_image = np.squeeze(np.squeeze(mr_fake.cpu(), axis=0), axis=0)
        return mr_image, mr_fake_image

    
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
                    original_mri_image, fake_mri_image = self.test_one(
                        batch=new_batch, 
                        save_path=save_path,
                        if_loop=if_loop,
                        idx=idx,
                    )
                    result_3d_z[i, :, :] = fake_mri_image
                result_3d_z.detach().numpy().tofile(f"{save_path}/{idx}.raw")
                fake_mri_volume = result_3d_z
                original_mri_volume = batch[1].squeeze().squeeze()
            else:   
                if self.dims == 2 and len(batch[0].shape) == 5:
                    if self.dataset_name == "MedicalDataset3D":
                        batch = (batch[0][:, :, :, self.image_size[1] // 2, :], batch[1][:, :, :, self.image_size[1] // 2, :])
                    else:
                        batch = (batch[0][:, :, self.image_size[0] // 2, :, :], batch[1][:, :, self.image_size[0] // 2, :, :])
                original_mri_volume, fake_mri_volume = self.test_one(
                    batch=batch, 
                    save_path=save_path,
                    if_loop=if_loop,
                    idx=idx,
                )
                fake_mri_volume.detach().numpy().tofile(f"{save_path}/{idx}.raw")

            if not if_loop:
                slices = []
                for volume in [original_mri_volume, fake_mri_volume]:
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
        self.parser.add_argument('--version', type=str, default=None)
        self.parser.add_argument('--in_channels', type=int, default=2)
        self.parser.add_argument('--double_channels', type=int, choices=[0, 1], default=1)
        self.parser.add_argument('--inner_channel', type=int, default=32)
        self.parser.add_argument('--use_vae', type=int, choices=[0, 1], default=0)
        self.parser.add_argument('--vae_weight_file', type=str, default="../VAE/weight_file/vqvae_3d_v2/best.pth")
        self.parser.add_argument('--vae_embedding_dims', type=int, default=2)
        self.parser.add_argument('--dims', type=int, choices=[2, 3], default=3)
        self.parser.add_argument('--use_collate', type=int, choices=[0, 1], default=0)
        self.parser.add_argument('--sample_method', type=str, choices=["ddpm", "ddim"], default="ddpm")
        self.parser.add_argument('--if_loop', type=int, choices=[0, 1], default=0)
        self.parser.add_argument('--if_best', type=int, choices=[0, 1], default=1)
        self.parser.add_argument('--generate_concatenated_volume', type=int, choices=[0, 1], default=0)
        
    def parse_args(self):
        args = self.parser.parse_args()
        args.use_vae = bool(int(args.use_vae))
        args.use_collate = bool(int(args.use_collate))
        args.if_loop = bool(int(args.if_loop))
        args.if_best = bool(int(args.if_best))
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
