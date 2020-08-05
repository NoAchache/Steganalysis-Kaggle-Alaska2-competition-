def flops(k, ch_in, ch_out, w, h, s=1):
    return (k * k + (k * k - 1)) * ch_in * ch_out * w * h / s

def SRM_additional_flops(img_size):
    'Additional FLOPS compared to the use of a simple efficient net'
    tot_flops = 0
    # 17 3*3 filters
    tot_flops += flops(3, 3, 17, img_size, img_size)
    # 13 5*5 filters
    tot_flops += flops(5, 3, 13, img_size, img_size)

    # First layer of efficient net now has 30 channels instead of 3. Since the number of FLOPS is proportional to the
    # number of input channels, we can consider 30 - 3 = 27 channels to get the number of extra FLOPS.
    # It outputs 32 channels and has a stride of 2.
    tot_flops += flops(3, 27, 32, img_size, img_size, 2)

    return tot_flops

def display_ratio(img_size):
    extra_flops = SRM_additional_flops(img_size)
    eff_net_b2_FLOPS = 1e9  # c.f. efficient_net paper
    ratio = extra_flops/eff_net_b2_FLOPS
    print(ratio)

if __name__ == '__main__':
    img_size = 224  # Same img shape as in the efficient net paper to compare with the number of FLOPS they computed
    display_ratio(img_size)


#########################################################################

# Efficient net paper: Tan, M., & Le, Q. V. (2019). Efficientnet: Rethinking model scaling for convolutional neural
# networks. arXiv preprint arXiv:1905.11946.
