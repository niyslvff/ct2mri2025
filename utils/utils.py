import matplotlib.pyplot as plt
import os
import numpy as np
import torch
import datetime
import random
from tqdm import tqdm

import networks_init_utils
import shutil
import csv


def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def remkdirs(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        os.makedirs(path)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CustomStepLR:
    def __init__(
        self,
        optimizer,
        step_size=1,
        lr_0=0.0002,
        gamma=(0.1, 0.5, 0.1, 0.2),
        last_epoch=-1,
        epochs=None,
    ):
        if epochs is None:
            epochs = [10, 20, 200, 500]
        assert len(epochs) == len(list(gamma)), "custom-stepLr wrong length."
        self.gamma = list(gamma)
        self.lrs = [lr_0]
        for i in range(len(epochs)):
            lr_0 = lr_0 * self.gamma[i]
            self.lrs.append(lr_0)
        print(self.lrs)
        self.optimizer = optimizer
        self.last_epoch = last_epoch
        self.step_size = step_size

        self.epochs = epochs
        for i in range(len(epochs)):
            if last_epoch < epochs[i]:
                break
            for group in self.optimizer.param_groups:
                group["lr"] = self.lrs[i + 1]
        print(
            "start epoch is {:d}\ninitial learning rate is {:.6f}".format(
                last_epoch, self.optimizer.param_groups[0]["lr"]
            )
        )

    def get_lr(self, gamma):
        if (self.last_epoch == 0) or (self.last_epoch % self.step_size != 0):
            return [group["lr"] for group in self.optimizer.param_groups]

        return [group["lr"] * gamma for group in self.optimizer.param_groups]

    def step(self):
        self.last_epoch += 1

        if self.last_epoch in self.epochs:
            idx = self.epochs.index(self.last_epoch)
            lrs = self.get_lr(self.gamma[idx])
            for param, lr in zip(self.optimizer.param_groups, lrs):
                param["lr"] = lr


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def load_optimizer(opt, file_path, auto_lr=False):
    base_lr = opt.param_groups[0]["lr"]
    opt.load_state_dict(file_path)
    if not auto_lr:
        set_lr(opt, base_lr)
    for state in opt.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.cuda()


def load_checkpoint(
    weight_file, model, optimizer, csv_dir=None, from_pth=False, auto_lr=False
):
    if from_pth:
        if not os.path.exists(weight_file):
            print("Weight file not exist!")
            raise "Error"
        checkpoint = torch.load(weight_file)
        # csv_file = open(csv_file, 'a', newline='')
        # writer = csv.writer(csv_file)
        model.load_state_dict(checkpoint["model"])
        load_optimizer(optimizer, checkpoint["optimizer"], auto_lr)

        best_epoch = checkpoint["best_epoch"]
        best_loss = checkpoint["best_loss"]
        start_epoch = checkpoint["epoch"] + 1
        print("Start from loss: {:.6f}.\n".format(checkpoint["loss"]))
    else:
        # model.weights_init()
        networks_init_utils.init_weights(model, init_type="normal")
        if csv_dir:
            for i in range(1000):
                with open(csv_dir + f"{i}.csv", "w", newline="") as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerow(("loss",))
        start_epoch = 0
        best_epoch = 0
        best_loss = 9999999

    return start_epoch, best_epoch, best_loss


def save_checkpoint(
    model,
    optimizer,
    epoch,
    epoch_losses,
    best_epoch,
    best_loss,
    outputs_dir,
    target_loss=None,
    latest_step=1,
):
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "loss": epoch_losses.avg,
        "best_loss": best_loss,
        "best_epoch": best_epoch,
    }
    if epoch % latest_step == 0:
        torch.save(state, outputs_dir + f"latest.pth")
    # csv_writer.writerow((state['epoch'], state['loss']))
    if target_loss is not None and epoch_losses.avg < target_loss:
        torch.save(
            state, outputs_dir + "{:d}_{:.4f}.pth".format(epoch, epoch_losses.avg)
        )

    if epoch_losses.avg < best_loss:
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "loss": epoch_losses.avg,
            "best_loss": best_loss,
            "best_epoch": best_epoch,
        }
        best_epoch = epoch
        best_loss = epoch_losses.avg
        state["best_epoch"] = best_epoch
        state["best_loss"] = best_loss
        torch.save(state, outputs_dir + f"best.pth")
    return best_epoch, best_loss


def showSlices(slices, rows, save_path=None):
    """
    create time: 2023/9/30
    """
    columns = (
        len(slices) // rows if len(slices) % rows == 0 else len(slices) // rows + 1
    )
    fig, axes = plt.subplots(rows, columns)
    if rows == 1:
        for i, slice in enumerate(slices[:]):
            axes[i].imshow(slice.T, cmap="gray", origin="lower")
    else:
        for row in range(rows):
            if row < rows - 1:
                for i, slice in enumerate(slices[row * columns : (row + 1) * columns]):
                    # axes[row][i].imshow(slice.T, cmap='gray', origin='lower')
                    axes[row][i].imshow(slice)
            else:
                for i, slice in enumerate(slices[row * columns :]):
                    # axes[row][i].imshow(slice.T, cmap='gray', origin='lower')
                    axes[row][i].imshow(slice)
    plt.savefig(save_path if save_path else "test.png")
    plt.close()


def showSlicesOneRaw(slices, words):
    _, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices[:]):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")
    plt.title(words)
    plt.show()


def show_t_loss_lists(t_loss_lists, timesteps=1000):
    # assert t < timesteps, "t must be less than timesteps."
    # assert len(t_loss_lists[t]) == 0, "list is empty."
    t = 0
    for i in range(len(t_loss_lists)):
        if len(t_loss_lists[i]) > 1:
            t = i
            break
    # 生成一些示例数据
    t_loss_list = t_loss_lists[t]
    print(f"show t{t} loss trend, num of t{t} loss is {len(t_loss_list)}.")
    x = list(range(len(t_loss_list)))
    y = t_loss_list
    # 绘制原始曲线
    plt.plot(x, y, label="Data")

    # 添加图例和标签
    plt.legend()
    plt.title("loss Trendline")
    plt.xlabel("X")
    plt.ylabel("loss")

    # 显示图形
    plt.show()


def saveDiffModel(
    epoch, step, model, opt, dataSavePath="../results/brain", save_folder_name="v0"
):
    """
    create time: 2023/10/1
    """

    fileName = "%s/Diffusion/%s/epoch%d_step%d.pth" % (
        dataSavePath,
        save_folder_name,
        epoch,
        step,
    )
    if not os.path.exists(f"{dataSavePath}/Diffusion/{save_folder_name}"):
        os.makedirs(f"{dataSavePath}/Diffusion/{save_folder_name}")

    torch.save({"diff": model.state_dict(), "opt": opt.state_dict()}, fileName)


def saveDiffRawFile10(
    epoch,
    step,
    volume,
    dataSavePath="../results/brain",
    save_folder_name="v0",
    original_img_size=(160, 224, 160),
    crop_rate=(1, 1, 1),
):
    """
    create time: 2023/10/1
    """
    fileName = "%s/Diffusion/%s/epoch%d_step%d.raw" % (
        dataSavePath,
        save_folder_name,
        epoch,
        step,
    )
    if not os.path.exists(f"{dataSavePath}/{save_folder_name}"):
        os.makedirs(f"{dataSavePath}/{save_folder_name}")
    volume = volume.view(
        int(crop_rate[0] * original_img_size[0]),
        int(crop_rate[1] * original_img_size[1]),
        int(crop_rate[2] * original_img_size[2]),
    )
    # copy tensor from gpu to cpu.
    volume = volume.cpu()
    # convert tensor to numpy ndarray.
    volume = volume.detach().numpy()
    volume.astype("float32").tofile(fileName)


def visualization_3d(file):
    data = np.fromfile(file, dtype=np.float32).reshape(160, 224, 160)
    data = np.clip(data, -1.0, 1.0)
    x, y, z, c = [], [], [], []
    for i in range(160):
        for j in range(160):
            for k in range(160):
                # print(data[i, j, k])
                if data[i, j, k] > 0.1:
                    x.append(i)
                    y.append(j)
                    z.append(k)
                    c.append(data[i, j, k])
                else:
                    continue
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(x, y, z, c=c)
    plt.show()


def save_3d_prediction(save_path, prefix, data, idx):
    prefix = prefix.split("/")[2]
    base_root = "%s/%s" % (save_path, prefix)
    mkdirs(base_root)

    uid = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(0, 60)}"
    filename = "%s/%s.%s.raw" % (base_root, idx, uid)
    volume = data.view(data.shape)
    # copy tensor from gpu to cpu.
    volume = volume.cpu()
    # convert tensor to numpy ndarray.
    volume = volume.detach().numpy()
    volume.astype("float32").tofile(filename)


def raw2png(path):
    data = np.fromfile(path, dtype=np.float32)
    data = data.reshape((224, 224))
    data.astype("float32").tofile("./test.png")
    print(data.shape)


def see_shape(path):
    data = np.fromfile(path, dtype=np.float32)
    print(f"origin length: {data.shape[0]}")
    less_length = int(np.sqrt(data.shape[0] // 3))
    print(f"cropped sqrt length: {less_length}")
    data = data[: less_length**2 * 3]
    print(data.shape)
    data.tofile(path)


def t_histogram(image_num, epochs, T):
    """
    see the histogram of timesteps.
    imitate 1 batch size for 180 * 0.8 * 4 times data loading and 250 epochs.
    :param timesteps:
    :param T:
    :return:
    """
    t_list = []

    with tqdm(total=(image_num * epochs)) as tq:
        for i in range(epochs):
            for j in range(image_num):
                t = torch.randint(0, T, (1,)).item()

                t_list.append(t)
                tq.update(1)
    plt.hist(t_list, bins=1000, color="blue", alpha=0.7, range=(0, T))

    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.title("Image Histogram")
    plt.savefig("histogram")


def volume_3d_visualization(img_path, img_shape=(256, 256, 256)):
    img_array = np.fromfile(img_path, dtype=np.float32).reshape(img_shape)
    # img_array = np.array([[[1, 0.5], [2, 0.1]], [[3, 0.2], [0.5, 1]]])

    # 获取图像的通道数、高度和宽度
    num, height, width = img_array.shape

    x_coords, y_coords, z_coords = np.meshgrid(
        np.arange(width), np.arange(height), np.arange(num), indexing="ij"
    )

    # 将图像坐标转换为点云数据
    point_cloud_data = np.column_stack(
        (x_coords.ravel(), y_coords.ravel(), z_coords.ravel())
    )
    c_values = img_array.reshape(-1)
    threshold = np.mean(c_values)

    # 绘制散点图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        point_cloud_data[:, 0],
        point_cloud_data[:, 1],
        point_cloud_data[:, 2],
        c=c_values,
        s=1,
        alpha=(c_values >= threshold).astype(float),
    )

    # 设置图表标题和坐标轴标签
    ax.set_title("Pixel Distribution of 3D Image")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_zlabel("Z Coordinate")

    # 显示图表
    plt.show()


def data_distribution_cloud(img_path, img_shape=(256, 256, 256)):
    pixel_values = np.fromfile(img_path, dtype=np.float32)

    # 将像素值和位置转换为点云数据
    point_cloud_data = np.column_stack((pixel_values, np.arange(len(pixel_values))))

    # 绘制散点图
    plt.scatter(point_cloud_data[:, 0], point_cloud_data[:, 1], s=1, alpha=0.5)

    # 设置图表标题和坐标轴标签
    plt.title("Pixel Value Scatter Plot")
    plt.xlabel("Pixel Value")
    plt.ylabel("Position in 1D Vector")

    # 显示图表
    plt.show()


def volume_histogram(img_path, name):
    plt.figure()
    pixel_values = np.fromfile(img_path, dtype=np.float32)
    plt.hist(pixel_values, bins=500, color="blue")

    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.title(f"{name}: Image Histogram")
    plt.show()


if __name__ == "__main__":
    # visualization_3d('crop/brain/ct/ct_1.raw')

    # raw2png("base_diffusion/prediction/CSAUNet_20231107_193057_47.raw")
    # see_shape("base_diffusion/CSAUNet_20231107_193057_47.raw")

    # t_histogram(image_num=int(180 * 0.8), epochs=2000, T=1000)
    volume_3d_visualization("./crop/brain1/mr/mr_0.raw")
    # data_distribution_cloud('./crop/brain1/ct/ct_0.raw')
    # volume_histogram('./crop/brain/ct/ct_0.raw', 'ct, MaxMinNormalization')
    # volume_histogram('./crop/brain2/ct/ct_0.raw', 'ct2, MaxMinNormalization')
    # volume_histogram('./crop/brain/mr/mr_0.raw', 'mri, MaxMinNormalization')
    # volume_histogram('./crop/brain2/mr/mr_0.raw', 'mri2, MaxMinNormalization')
