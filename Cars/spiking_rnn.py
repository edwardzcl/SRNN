import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
thresh = 1.0 # neuronal threshold
lens = 0.5 # hyper-parameters of approximate function
decay = 1.0 # decay constants
num_classes = 2
batch_size  = 100
learning_rate = 1e-3
num_epochs = 10 # max epoch
# define approximate firing function
class ActFun(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.ge(thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input - thresh) < lens
        return grad_input * temp.float()

act_fun = ActFun.apply
# membrane potential update
def mem_update(ops1, ops2, x, mem, spike):
    mem = mem * decay * (1. - spike) + ops1(x) + ops2(spike) / 10. #ops2(mem)
    spike = act_fun(mem) # act_fun : approximation firing function
    return mem, spike

###量化层###

class STERound(torch.autograd.Function):
    """
    A class to implement STE round function.
    Override backward gradient with 1.
    """

    @staticmethod
    def forward(ctx, x):
        #self.saved_for_backward = [x, k]
        #n = torch.tensor(2**k)
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad):
        return grad



class Quantize_Spike(nn.Module):
    """
    Weight quantization operation.
    Do rounding and clamping.
    Use STE trick to make gradient continuous.
    Weight is always signed.
    """

    def __init__(self, k, B):
        super().__init__()
        self.quant_level = k
        self.quant_bound = B


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = float(2**self.quant_level)
        return STERound.apply(torch.clamp(x, -self.quant_bound, self.quant_bound) * n) / n

###量化层###


class QuantLinear(nn.Linear):
    def __init__(
        self,
        input_dim = None,
        output_dim = None,
        W=3
    ):
        super(QuantLinear, self).__init__(
            input_dim,
            output_dim,
            False
        )
        #self.quant_inference = quant_inference
        self.weight_quantizer = WeightQuantizer(W=W)

    def forward(self, input):
        tnn_bin_weight = self.weight_quantizer(self.weight)
        output = F.linear(
            input,
            tnn_bin_weight
        )
        return output


class WeightQuantizer(nn.Module):
    def __init__(self, W=2):
        super(WeightQuantizer, self).__init__()
        self.W = W


    def ternary(self, input):
        output = Ternary.apply(input)
        return output

    def forward(self, input):

        output_fp = input.clone()
        # ************** W —— +-1、0 **************
        output, threshold = self.ternary(input)  # threshold(阈值)
        # **************** α(缩放因子) ****************
        #output_abs = torch.abs(output_fp)
        #mask_le = output_abs.le(threshold)
        #mask_gt = output_abs.gt(threshold)
        #output_abs[mask_le] = 0
        #output_abs_th = output_abs.clone()
        #output_abs_th_sum = torch.sum(output_abs_th, (3, 2, 1), keepdim=True)
        #mask_gt_sum = torch.sum(mask_gt, (3, 2, 1), keepdim=True).float()
        #alpha = output_abs_th_sum / mask_gt_sum  # α(缩放因子)
        # *************** W * α ****************
        #output = output * alpha  # 若不需要α(缩放因子)，注释掉即可

        return output


# ********************* 三值(+-1、0) ***********************
class Ternary(torch.autograd.Function):
    @staticmethod
    def forward(self, input):
        # **************** channel级 - E(|W|) ****************
        E = torch.mean(torch.abs(input))
        # **************** 阈值 ****************
        threshold = E * 0.7
        # ************** W —— +-1、0 **************
        output = torch.sign(
            torch.add(
                torch.sign(torch.add(input, threshold)),
                torch.sign(torch.add(input, -threshold)),
            )
        )
        return output, threshold

    @staticmethod
    def backward(self, grad_output, grad_threshold):
        # *******************ste*********************
        grad_input = grad_output.clone()
        return grad_input



# cnn_layer(in_planes, out_planes, stride, padding, kernel_size)
cfg_cnn = [(1, 32, 1, 1, 3),
           (32, 32, 1, 1, 3),]
# kernel size
cfg_kernel = [28, 14, 7]
# fc layer
cfg_fc = [128, 10]

# Dacay learning_rate
def lr_scheduler(optimizer, epoch, init_lr=0.1, lr_decay_epoch=50):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    if epoch % lr_decay_epoch == 0 and epoch > 1:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
    return optimizer

class SCNN(nn.Module):
    def __init__(self):
        super(SCNN, self).__init__()
        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[0]
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)
        self.recu_conv1 = nn.Conv2d(out_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)
        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[1]
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)
        self.recu_conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)

        self.fc1 = nn.Linear(cfg_kernel[-1] * cfg_kernel[-1] * cfg_cnn[-1][1], cfg_fc[0])
        self.recu_fc1 = nn.Linear(cfg_fc[0], cfg_fc[0])
        self.fc2 = nn.Linear(cfg_fc[0], cfg_fc[1])
        self.recu_fc2 = nn.Linear(cfg_fc[1], cfg_fc[1])

    def forward(self, input, time_window = 20):
        c1_mem = c1_spike = torch.zeros(batch_size, cfg_cnn[0][1], cfg_kernel[0], cfg_kernel[0], device=device)
        c2_mem = c2_spike = torch.zeros(batch_size, cfg_cnn[1][1], cfg_kernel[1], cfg_kernel[1], device=device)

        h1_mem = h1_spike = h1_sumspike = torch.zeros(batch_size, cfg_fc[0], device=device)
        h2_mem = h2_spike = h2_sumspike = torch.zeros(batch_size, cfg_fc[1], device=device)

        for step in range(time_window): # simulation time steps
            x = input > torch.rand(input.size(), device=device) # prob. firing

            c1_mem, c1_spike = mem_update(self.conv1, self.recu_conv1, x.float(), c1_mem, c1_spike)

            x = F.avg_pool2d(c1_spike, 2)

            c2_mem, c2_spike = mem_update(self.conv2, self.recu_conv2, x, c2_mem,c2_spike)

            x = F.avg_pool2d(c2_spike, 2)
            x = x.view(batch_size, -1)

            h1_mem, h1_spike = mem_update(self.fc1, self.recu_fc1, x, h1_mem, h1_spike)
            h1_sumspike = h1_sumspike + h1_spike
            h2_mem, h2_spike = mem_update(self.fc2, self.recu_fc2, h1_spike, h2_mem,h2_spike)
            h2_sumspike = h2_sumspike + h2_spike
            


        outputs = h2_sumspike / time_window
        return outputs


