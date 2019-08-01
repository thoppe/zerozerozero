# Works on python 3.6.4
# Using requirements listed


import sys, os, argparse

sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))

import torch
import cvbase as cvb
from torch.utils.data import DataLoader

from models import FlowNet2

# from dataset.FlowInfer import FlowInfer

import torch
import cv2
import numpy as np
import torch.utils.data


save_dest = 'data/flows'
os.system(f'mkdir -p {save_dest}')


class FlowInfer(torch.utils.data.Dataset):
    def __init__(self, list_file, size=None, isRGB=True, start_pos=0):
        super(FlowInfer, self).__init__()
        self.size = size
        txt_file = open(list_file, "r")
        self.frame1_list = []
        self.frame2_list = []
        self.output_list = []
        self.isRGB = isRGB

        for line in txt_file:
            line = line.strip(" ")
            line = line.strip("\n")

            line_split = line.split(" ")
            self.frame1_list.append(line_split[0])
            self.frame2_list.append(line_split[1])
            self.output_list.append(line_split[2])

        if start_pos > 0:
            self.frame1_list = self.frame1_list[start_pos:]
            self.frame2_list = self.frame2_list[start_pos:]
            self.output_list = self.output_list[start_pos:]
        txt_file.close()

    def __len__(self):
        return len(self.frame1_list)

    def __getitem__(self, idx):
        frame1 = cv2.imread(self.frame1_list[idx])
        frame2 = cv2.imread(self.frame2_list[idx])
        if self.isRGB:
            frame1 = frame1[:, :, ::-1]
            frame2 = frame2[:, :, ::-1]
        output_path = self.output_list[idx]

        frame1 = self._img_tf(frame1)
        frame2 = self._img_tf(frame2)

        frame1_tensor = torch.from_numpy(frame1).permute(2, 0, 1).contiguous().float()
        frame2_tensor = torch.from_numpy(frame2).permute(2, 0, 1).contiguous().float()

        return frame1_tensor, frame2_tensor, output_path

    def _img_tf(self, img):
        img = cv2.resize(img, (self.size[1], self.size[0]))

        return img


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pretrained_model_flownet2",
        type=str,
        default="./models/FlowNet2_checkpoint.pth.tar",
    )

    parser.add_argument("--img_size", type=list, default=(512, 1024, 3))
    parser.add_argument("--rgb_max", type=float, default=255.0)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument(
        "--data_list", type=str, default=None, help="Give the data list to extract flow"
    )
    parser.add_argument(
        "--frame_dir",
        type=str,
        default="data/frames/",
        help="Give the dir of the video frames and generate the data list to extract flow",
    )

    args = parser.parse_args()
    return args


def infer(args):
    assert args.data_list is not None or args.frame_dir is not None

    if args.frame_dir is not None:
        data_list = generate_flow_list(args.frame_dir)
        args.data_list = data_list

    with open(data_list) as FIN:
        F_SAVE = [line.split()[0] for line in FIN]

    device = torch.device("cuda:0")

    Flownet = FlowNet2(args, requires_grad=False)
    print("====> Loading", args.pretrained_model_flownet2)
    flownet2_ckpt = torch.load(args.pretrained_model_flownet2)
    Flownet.load_state_dict(flownet2_ckpt["state_dict"])
    Flownet.to(device)
    Flownet.eval()

    dataset_ = FlowInfer(args.data_list, size=args.img_size)
    dataloader_ = DataLoader(dataset_, batch_size=1, shuffle=False)
    # task_bar = ProgressBar(dataset_.__len__())

    for i, (f1, f2, output_path_) in enumerate(dataloader_):
        f1 = f1.to(device)
        f2 = f2.to(device)

        flow = Flownet(f1, f2)

        '''
        output_path = output_path_[0]

        output_file = os.path.dirname(output_path)
        if not os.path.exists(output_file):
            os.makedirs(output_file)
        '''

        flow_numpy = flow[0].permute(1, 2, 0).data.cpu().numpy()
        f_save = os.path.join(save_dest, os.path.basename(F_SAVE[i])) + '.npy'

        print(flow_numpy.shape)
        print(flow_numpy.mean())

        np.save(f_save, flow_numpy)
        

        #cvb.write_flow(flow_numpy, output_path)
        # task_bar.update()

    print("FlowNet2 Inference has been finished~!")
    print("Extracted Flow has been save in", output_file)

    return 


def generate_flow_list(frame_dir):
    dataset_root = os.path.dirname(frame_dir)
    video_root = frame_dir
    train_list = open(os.path.join(dataset_root, "video.txt"), "w")
    flow_list = open(os.path.join(dataset_root, "video_flow.txt"), "w")
    output_root = os.path.join(dataset_root, "Flow")

    img_total = 0
    video_id = os.path.basename(frame_dir)

    img_id_list = [x for x in os.listdir(video_root) if ".png" in x or ".jpg" in x]
    img_id_list.sort()
    img_num = len(img_id_list)
    train_list.write(video_id)
    train_list.write(" ")
    train_list.write(str(img_num))
    train_list.write("\n")
    img_total += img_num

    for i in range(img_num):
        if i + 1 < img_num:
            flow_list.write(os.path.join(video_root, img_id_list[i]))
            flow_list.write(" ")
            flow_list.write(os.path.join(video_root, img_id_list[i + 1]))
            flow_list.write(" ")
            flow_list.write(os.path.join(output_root, img_id_list[i][:-4] + ".flo"))
            flow_list.write("\n")

        if i - 1 >= 0:
            flow_list.write(os.path.join(video_root, img_id_list[i]))
            flow_list.write(" ")
            flow_list.write(os.path.join(video_root, img_id_list[i - 1]))
            flow_list.write(" ")
            flow_list.write(os.path.join(output_root, img_id_list[i][:-4] + ".rflo"))
            flow_list.write("\n")

    print("This Video has", img_total, "Images")
    train_list.close()
    flow_list.close()
    print(
        "The optical flow list has been generated:",
        os.path.join(dataset_root, "video_flow.txt"),
    )

    return os.path.join(dataset_root, "video_flow.txt")


def main():
    args = parse_args()
    infer(args)


if __name__ == "__main__":
    main()
