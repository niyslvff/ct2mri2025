import os
import sys
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append("..")

from torch.utils.data import DataLoader
import torch
from torchvision.transforms import Compose
import torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import Optional, TypeVar
Tensor = TypeVar("torch.Tensor")
import argparse
import datetime
import numpy as np

from tqdm import tqdm
from gaussian_diffusion import DDPMScheduler
from models.vqvae.vq_vae import VQVAE


class Train:
    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        self.dims = kwargs.get("dims")
        self.crop = kwargs.get("crop")
        self.epoch_num = kwargs.get("epoch_num")
        self.batch_size = kwargs.get("batch_size")
        self.in_channels = kwargs.get("in_channels")
        self.learning_rate = kwargs.get("learning_rate")
        self.timestep_amount = kwargs.get("timestep_amount")
        self.model_name = kwargs.get("model_name")
        self.version = kwargs.get("version")
        self.use_vae = kwargs.get("use_vae")
        self.vae_weight_file = kwargs.get("vae_weight_file")
        self.vae_embedding_dims = kwargs.get("vae_embedding_dims")
        self.image_size = kwargs.get("image_size")
        self.image_root = kwargs.get("image_root")
        self.dataset_name = kwargs.get("dataset_name")
        self.loss_function = kwargs.get("loss_function")
        self.inner_channels = kwargs.get("inner_channels")
        self.use_collate = kwargs.get("use_collate")
        self.use_condition = kwargs.get("use_condition")
        self.organ = kwargs.get("organ")
        self.create_time = kwargs.get("create_time")
        
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.average_loss_log = {"val": 0, "avg": 0, "sum": 0, "count": 0}
        self.current_epoch = 0
        self.best_epoch = 0
        self.current_loss = 99999
        self.best_loss = 99999
        
        self.image_size_2d = (224, 224) # MedicalDataset3D
        self.new_image_size = (168, 224) if self.crop else (84, 112)   # # (168, 224) or (84, 112)
        crop_transform = Compose([
            torchvision.transforms.RandomCrop(self.new_image_size),
        ])
        self.transform = Compose([
                torchvision.transforms.Resize(self.image_size_2d),
                torchvision.transforms.Lambda(lambda img: img if not self.crop else crop_transform(img)),
                torchvision.transforms.RandomHorizontalFlip(0.5),
        ])

        self.dataset = self.choose_dataset()
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=self.dataloader_collate_fn_2d if self.dims == 2 else None,
        )
        
        self.model = self.choose_model()
        self.vae_model = VQVAE(
            in_channels=1,
            out_channels=1,
            embedding_dim=self.vae_embedding_dims,
            num_embeddings=512,
            beta=0.25,
            dims=self.dims,
        )
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
        )
        
    def choose_dataset(self):
        from dataset import MedicalDataset3D, Raw3dDiffusionConcatenateDataset, MixedSize2DDataset, Raw3dDiffusionMajiDataset
        if self.dataset_name == "MedicalDataset3D":
            return MedicalDataset3D(
                image_root=self.image_root,
                image_size=self.image_size,
            )
        elif self.dataset_name == "concatenation":
            return Raw3dDiffusionConcatenateDataset(
                img_num=200,
                img_root_path=self.image_root,
                img_shape=self.image_size,
                is_val_set=False,
                use_crop=self.crop,
                transform=None,
            )
        elif self.dataset_name == "MixedSize2DDataset":
            return MixedSize2DDataset(
                file_path=self.image_root,
                crop=self.crop,
                is_val=False,
            )
        elif self.dataset_name == "Raw3dDiffusionMajiDataset":
            return Raw3dDiffusionMajiDataset(
                img_root_path=self.image_root,
                use_crop=self.crop,
                is_val=False,
                img_shape=self.image_size,
            )
        else:
            raise ValueError("不支持的数据集")
        
    def choose_model(self):
        if self.model_name.lower() == "vit":
            from models.vit import ViT
            return ViT(
                # self.image_size_2d,
                input_size=(84, 112) if self.crop else (168, 224),
                patch_size=8,
                in_channel=2,
                hidden_channels=256,
                out_channels=1,
            )
        elif self.model_name.lower() == "dit":
            from models.dit.models import DiT_models
            return DiT_models['DiT-S/8'](
                input_size=(224, 224), 
                in_channels=1,
                learn_sigma=False,
            )
        elif self.model_name.lower() == "sunet":
            if self.version.lower() == "v1":
                from models.sunet.sunet_v1 import UNet
            elif self.version.lower() == "v2":
                from models.sunet.sunet_v2 import UNet
            elif self.version.lower() == "v1_1":
                from models.sunet.sunet_v1_1 import UNet
            elif self.version.lower() == "v1_2":
                from models.sunet.sunet_v1_2 import UNet
            else:
                raise "wrong version."
            return UNet(
                in_channel=self.in_channels,
                out_channel=1,
                dims=self.dims,
                inner_channel=self.inner_channels,
            )
        else:
            raise "wrong model."
            

    def reset_average_loss_log(self):
        r"""重置loss记录。
        """
        for key, _ in self.average_loss_log.items():
            self.average_loss_log[key] = 0

    def update_average_loss_log(self, val, n=1):
        r"""更新loss记录。
        """
        self.average_loss_log["val"] = val
        self.average_loss_log["sum"] += val * n
        self.average_loss_log["count"] += n
        self.average_loss_log["avg"] = (
            self.average_loss_log["sum"] / self.average_loss_log["count"]
        )

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

        return original_ct_tensors, original_mr_tensors, transformed_ct_tensors, transformed_mr_tensors
    
    def crop_collate_fn_3d(self, batch: tuple[Tensor, Tensor]):
        seed = torch.random.seed()
        ct_batch, mri_batch = [], []
        random_depth = np.random.randint(0, self.image_size[0] // 2)
        if self.batch_size == 1:
            new_batch = (batch[0][:, :, random_depth:random_depth + self.image_size[0] // 2, :, :], batch[1][:, :, random_depth:random_depth + self.image_size[0] // 2, :, :])
            ct, mri = new_batch
            torch.random.manual_seed(seed)
            new_ct = torch.stack([self.transform(slice) for slice in ct[0][0]]).unsqueeze(0).unsqueeze(0)
            torch.random.manual_seed(seed)
            new_mri = torch.stack([self.transform(slice) for slice in mri[0][0]]).unsqueeze(0).unsqueeze(0)
            return new_ct, new_mri
        else:
            new_batch = [(tensor[0][:, random_depth:random_depth + self.image_size[0] // 2, :, :], tensor[1][:, random_depth:random_depth + self.image_size[0] // 2, :, :]) for tensor in batch]
            for tensors in new_batch:
                ct, mri = tensors
                new_ct, new_mri = [], []
                torch.random.manual_seed(seed)
                for slice in ct[0]:
                    torch.random.manual_seed(seed)
                    new_ct.append(self.transform(slice))
                new_ct = torch.stack(new_ct).unsqueeze(0)
                torch.random.manual_seed(seed)
                for slice in mri[0]:
                    torch.random.manual_seed(seed)
                    new_mri.append(self.transform(slice))
                new_mri = torch.stack(new_mri).unsqueeze(0)
                ct_batch.append(new_ct);mri_batch.append(new_mri)
            return torch.stack(ct_batch), torch.stack(mri_batch)
        # print(new_transformed_ct_tensors.shape)
    
    def compute_loss(self, noise_mr_pred: Tensor, noise_mr_real: Tensor) -> Tensor:
        if self.loss_function == "l1":
            loss = F.l1_loss(noise_mr_pred, noise_mr_real)
        elif self.loss_function == "mse":
            loss = F.mse_loss(noise_mr_pred, noise_mr_real)
        elif self.loss_function == "double":
            adv_loss = F.l1_loss(noise_mr_pred, noise_mr_real)
            noise_mr_pred_slices_consistence = torch.zeros(noise_mr_real.shape).to(self.device)
            noise_mr_real_slices_consistence = torch.zeros(noise_mr_real.shape).to(self.device)
            for z_slice_i in range(1, noise_mr_real.shape[2]):
                noise_mr_pred_slice_consistence = noise_mr_pred[:, :, z_slice_i, :, :] - noise_mr_pred[:, :, z_slice_i - 1, :, :]
                noise_mr_pred_slices_consistence[:, :, z_slice_i, :, :] = noise_mr_pred_slice_consistence.unsqueeze(0)

                noise_mr_real_slice_consistence = noise_mr_real[:, :, z_slice_i, :, :] - noise_mr_real[:, :, z_slice_i - 1, :, :]
                noise_mr_real_slices_consistence[:, :, z_slice_i, :, :] = noise_mr_real_slice_consistence.unsqueeze(0)
            consistent_loss = F.mse_loss(noise_mr_pred_slices_consistence, noise_mr_real_slices_consistence)
            loss = 0.8 * adv_loss + 0.2 * consistent_loss
        return loss

    def __call__(self):
        print("start training.")
        if self.create_time is None:
            self.create_time = datetime.datetime.now().strftime("%Y%m%d")
        self.checkpoints_root_path = f"./{self.model_name}/weight_file/{self.organ}/{self.model_name}.{self.version}.{self.dims}d{'.latent' if self.use_vae else ''}.{self.create_time}"
        if not os.path.exists(self.checkpoints_root_path):
            os.makedirs(self.checkpoints_root_path)
        
        self.model.to(self.device)
        self.load_checkpoints()
        self.model.train()
        
        if self.use_vae:
            self.vae_model.to(self.device)
            vae_checkpoints = torch.load(self.vae_weight_file)
            try:
                self.vae_model.load_state_dict(vae_checkpoints['model'])
            except:
                self.vae_model.load_state_dict(vae_checkpoints['model_parameters'])
            self.vae_model.eval()
        
        gaussain_scheduler = DDPMScheduler(
            num_train_timesteps=self.timestep_amount
        )
        for epoch in range(self.current_epoch, self.epoch_num):
            with tqdm(total=self.dataset.__len__() - self.dataset.__len__() % self.batch_size) as tq:
                tq.set_description(f"epoch: {epoch}/{self.epoch_num} ")
                self.reset_average_loss_log()
                for batch in self.dataloader:
                    # （原始ct张量, 原始mri张量, 变形后的ct张量, 变形后的mri张量）
                    # self.save_image(batch[0], 1)
                    # self.save_image(batch[1], 2)
                    # self.save_image(batch[2], 3)
                    # self.save_image(batch[3], 4)
                    # batch = self.crop_collate_fn_3d(batch)

                    if self.model_name == "dit":
                        original_ct_tensors, original_mr_tensors, transformed_ct_tensors, transformed_mr_tensors = batch
                    else:
                        transformed_ct_tensors, transformed_mr_tensors = batch
                    transformed_ct_tensors, transformed_mr_tensors = transformed_ct_tensors.to(self.device), transformed_mr_tensors.to(self.device)
                    # self.save_image(transformed_ct_tensors, 3)
                    # self.save_image(transformed_mr_tensors, 4)
                    # break
                
                    if self.use_vae:
                        with torch.no_grad():
                            transformed_ct_tensors = self.vae_model.encode(transformed_ct_tensors)[0]
                            transformed_mr_tensors = self.vae_model.encode(transformed_mr_tensors)[0]
                            # 取通道0
                            rand_channel = np.random.randint(0, self.vae_embedding_dims)
                            transformed_ct_tensors = transformed_ct_tensors[:, rand_channel, :, :, :].unsqueeze(1)
                            transformed_mr_tensors = transformed_mr_tensors[:, rand_channel, :, :, :].unsqueeze(1)

                    self.optimizer.zero_grad()
                    t = torch.randint(0, self.timestep_amount, (self.batch_size,), device=self.device).long()
                    # 使用前线传播得到带噪声的mri图片和添加的噪声
                    transformed_mr_noised_image, mr_real_noise = gaussain_scheduler.forward_diffusion_sample(transformed_mr_tensors, t, device=self.device)
                    # 如果输入channel数为2则模型输入包括 (带噪声图, 条件图, 时间步）
                    # 如果输入channel数为1则模型输入包括 (带噪声图, 时间步)
                    mr_pred_noise = self.model(transformed_mr_noised_image, transformed_ct_tensors, t) if self.use_condition else self.model(transformed_mr_noised_image, t)
                    # print(f"pred: {mr_pred_noise.shape}, real: {mr_real_noise.shape}")
                    loss = self.compute_loss(mr_pred_noise, mr_real_noise)
                    self.update_average_loss_log(loss.item(), self.batch_size)
                    loss.backward()
                    self.optimizer.step()

                    tq.set_postfix(
                        loss="{:.6f}".format(self.average_loss_log["avg"]),
                        best_epoch=self.best_epoch,
                        best_loss=self.best_loss,
                    )
                    tq.update(self.batch_size)
            self.save_checkpoints(current_epoch=epoch, current_loss=self.average_loss_log["avg"])
            # break
        print("best epoch: {}, loss: {:.5f}".format(self.best_epoch, self.best_loss))

    def load_checkpoints(self):
        r"""加载权重。
        """
        checkpoints_root_path = self.checkpoints_root_path
        checkpoints_path = checkpoints_root_path + "/latest.pth"
        if os.path.exists(checkpoints_path):
            checkpoints_file = torch.load(checkpoints_path)
            self.best_epoch = checkpoints_file["best_epoch"]
            self.current_epoch = checkpoints_file["current_epoch"]
            self.best_loss = checkpoints_file["best_loss"]
            self.current_loss = checkpoints_file["current_loss"]
            self.model.load_state_dict(checkpoints_file["model_parameters"])
            base_learning_rate = self.optimizer.param_groups[0]["lr"]
            self.optimizer.load_state_dict(checkpoints_file["optimizer_parameters"])
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(self.device)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = base_learning_rate

            print(f"Start from epoch {self.current_epoch}, loss {self.current_loss}.")
            print(
                f"Best epoch {self.best_epoch}, best loss {self.best_loss}.\n", 20 * "="
            )
        else:
            print("Start from zero.")

    def save_checkpoints(self, current_loss: float, current_epoch: int):
        r"""保存权重。
        """
        checkpoints_latest_path = self.checkpoints_root_path + "/latest.pth"
        checkpoints_best_path = self.checkpoints_root_path + "/best.pth"
        save_best_model = False

        self.current_epoch = current_epoch
        self.current_loss = current_loss
        if self.best_loss >= self.current_loss:
            self.best_loss = self.current_loss
            self.best_epoch = self.current_epoch
            save_best_model = True
        state = {
            "model_parameters": self.model.state_dict(),
            "optimizer_parameters": self.optimizer.state_dict(),
            "current_epoch": self.current_epoch,
            "current_loss": self.current_loss,
            "best_epoch": self.best_epoch,
            "best_loss": self.best_loss,
        }
        torch.save(state, checkpoints_latest_path)
        if save_best_model:
            torch.save(state, checkpoints_best_path)

    @staticmethod
    def save_image(current_tensor, idx: int = None):
        current_image = current_tensor[0][0].detach().cpu()
        print("save_image:", current_image.shape)
        plt.figure()
        if len(current_image) == 3:
            plt.imshow(current_image[current_image.shape[0] // 2, :, :])
        else:
            plt.imshow(current_image)
        plt.savefig(f"train_model_{idx}.png")


class BaseCommand:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="基础命令行解析器")
        self.add_base_arguments()

    def add_base_arguments(self):
        # 添加基础参数
        self.parser.add_argument('--organ', type=str, choices=["brain", "pelvis"], default="brain")
        self.parser.add_argument('--depth', type=int, default=160)
        self.parser.add_argument('--height', type=int, default=224)
        self.parser.add_argument('--width', type=int, default=168)
        self.parser.add_argument('--image_root', type=str, default="../crop/preprocess_globalNormAndEnContrast")
        self.parser.add_argument('--dataset_name', type=str, default="MedicalDataset3D")
        self.parser.add_argument('--batch_size', type=int, default=1)
        self.parser.add_argument('--dims', type=int, default=3)
        self.parser.add_argument('--epoch_num', type=int, default=8000)
        self.parser.add_argument('--timestep_amount', type=int, default=1000)
        self.parser.add_argument('--learning_rate', type=float, default=0.00002)
        self.parser.add_argument('--crop', type=int, choices=[0, 1], default=0)
        self.parser.add_argument('--model_name', type=str, default='sunet')
        self.parser.add_argument('--inner_channels', type=int, default=32)
        self.parser.add_argument('--in_channels', type=int, default=1)
        self.parser.add_argument('--version', type=str, default=None)
        self.parser.add_argument('--loss_function', type=str, default="l1")
        self.parser.add_argument('--use_collate', type=int, choices=[0, 1], default=0)
        self.parser.add_argument('--use_condition', type=int, choices=[0, 1], default=1)
        self.parser.add_argument('--use_vae', type=int, choices=[0, 1], default=0)
        self.parser.add_argument('--vae_weight_file', type=str, default="./vqvae/weight_file/brain/v1/best.pth")
        self.parser.add_argument('--vae_embedding_dims', type=int, default=2)
        self.parser.add_argument('--create_time', type=str, default=None)

    def parse_args(self):
        args = self.parser.parse_args()
        args.crop = bool(int(args.crop))
        args.use_vae = bool(int(args.use_vae))
        args.use_collate = bool(int(args.use_collate))
        args.use_condition = bool(int(args.use_condition))
        args.image_size = (args.depth, args.height, args.width)
        return args
        

if __name__ == "__main__":
    command_parse = BaseCommand()
    args = command_parse.parse_args()
    print(args)
    
    print("是否开始测试？(1/0)")
    if int(input()) == 1:
        train = Train(**vars(args))
        train()
