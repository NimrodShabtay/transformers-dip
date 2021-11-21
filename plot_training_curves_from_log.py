import os
import matplotlib.pyplot as plt
import sys
import re


def parse_logs_to_lists(logs_path):
    with open(logs_path, 'r') as f:
        lines = f.readlines()

    loss_list = []
    psnr_noise_list = []
    psnr_gt_list = []
    iter_list = []

    for line in lines:
        if 'Iteration' not in line:
            continue
        cur_iter, cur_loss, cur_psnr_noisy, cur_psnt_gt, _ = re.findall(r"[-+]?\d*\.\d+|\d+",
                                                                        line[line.find('INFO'):-1])
        loss_list.append(float(cur_loss))
        psnr_noise_list.append(float(cur_psnr_noisy))
        psnr_gt_list.append(float(cur_psnt_gt))
        iter_list.append(int(cur_iter))

    return loss_list, psnr_noise_list, psnr_gt_list, iter_list


if __name__ == '__main__':
    log_file_path = sys.argv[1]
    dst_path = sys.argv[2]

    plt.figure(figsize=(7, 7))
    loss_list, psnr_noise_list, psnr_gt_list, iter_list = parse_logs_to_lists(log_file_path)
    psnr_gt_hanlde = plt.plot(iter_list, psnr_gt_list, label='psnr_gt')
    psnr_noise_handle = plt.plot(iter_list, psnr_noise_list, label='psnr_noise_gt')
    plt.legend()
    plt.title('PSNR-GT vs PSNR-NOISE-GT')
    plt.savefig(os.path.join(dst_path, 'psnrs.png'))
    plt.show()
