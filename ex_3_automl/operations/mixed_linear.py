import torch
import torch.nn as nn
import torch.nn.functional as F


class MixedLinear(nn.Module):
    def __init__(self, input_dim, output_dim,  max_out_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.max_out_dim = max_out_dim

    def sample_weights_and_bias(self, linear_layer):
        weight = linear_layer.weight[:self.output_dim, :self.input_dim]
        bias = linear_layer.bias[:self.output_dim]
        return weight, bias

    def forward(self, x, linear_layer):
        weight, bias = self.sample_weights_and_bias(linear_layer)
        out = F.linear(x[:, :, :self.input_dim], weight, bias)
        # pad output
        if out.shape[-1] != self.max_out_dim:
            out = F.pad(out, (0, self.max_out_dim -
                        out.shape[-1]), "constant", 0)
        return out


class MixedLinearV2Emb(nn.Module):
    def __init__(self, emb_dim_list, max_out_dim, linear_layer):
        super().__init__()
        self.emb_dim_list = emb_dim_list
        self.linear_layer = linear_layer
        self.max_out_dim = max_out_dim

    def sample_weights_and_bias(self, input_dim, output_dim, linear_layer):
        weight = linear_layer.weight[:output_dim, :input_dim]
        bias = linear_layer.bias[:output_dim]
        return weight, bias

    def forward(self, x, weights, use_argmax=False):
        if use_argmax:
            weights = torch.tensor(weights)
            weights_max = torch.argmax(weights, dim=-1)
            weight, bias = self.sample_weights_and_bias(
                self.emb_dim_list[weights_max], self.emb_dim_list[weights_max], self.linear_layer)
            # pad weights and bias
            weight = weights[weights_max]*F.pad(weight, (0, self.max_out_dim -
                            weight.shape[-1], 0, self.max_out_dim - weight.shape[-1]), "constant", 0)
            bias = weights[weights_max]*F.pad(bias, (0, self.max_out_dim -
                            bias.shape[-1]), "constant", 0)
            out = F.linear(x, weight, bias)
        else:
            weights_mix = 0
            bias_mix = 0
            # print("X shape: ", x.shape)
            for i in range(len(self.emb_dim_list)):
                weight, bias = self.sample_weights_and_bias(
                self.emb_dim_list[i], self.emb_dim_list[i], self.linear_layer)
                # pad weights and bias
                weight = F.pad(weight, (0, self.max_out_dim -
                           weight.shape[-1], 0, self.max_out_dim - weight.shape[-1]), "constant", 0)
                bias = F.pad(bias, (0, self.max_out_dim -
                         bias.shape[-1]), "constant", 0)
                weights_mix += weights[i]*weight
                bias_mix += weights[i]*bias
                out = F.linear(x, weights_mix, bias_mix)

        return out


class MixedLinearV2(nn.Module):
    def __init__(self, input_dim_list,  output_dim_list, max_out_dim, linear_layer, reverse=False):
        super().__init__()
        self.input_dim_list = input_dim_list
        self.output_dim_list = output_dim_list
        self.linear_layer = linear_layer
        self.max_in_dim = max(self.input_dim_list)
        self.max_out_dim = max(self.input_dim_list)*max(self.output_dim_list)
        self.reverse = reverse
        if reverse:
            self.max_out_dim = max(self.input_dim_list)
            self.max_in_dim = max(self.input_dim_list) * \
                max(self.output_dim_list)

    def sample_weights_and_bias(self, input_dim, output_dim, linear_layer):
        weight = linear_layer.weight[:output_dim, :input_dim]
        bias = linear_layer.bias[:output_dim]
        return weight, bias

    def forward(self, x, weights, use_argmax=False):
        if use_argmax:
            weights = torch.tensor(weights)
            weights_max = torch.argmax(weights, dim=-1)
            input_dim_argmax_id = weights_max%len(self.output_dim_list)
            output_dim_argmax_id = weights_max%len(self.input_dim_list)
            if self.reverse:
                weight, bias = self.sample_weights_and_bias(
                    self.output_dim_list[output_dim_argmax_id]*self.input_dim_list[input_dim_argmax_id], self.input_dim_list[input_dim_argmax_id], self.linear_layer)
            else:
                weight, bias = self.sample_weights_and_bias(
                    self.input_dim_list[input_dim_argmax_id], self.output_dim_list[output_dim_argmax_id]*self.input_dim_list[input_dim_argmax_id], self.linear_layer)
            # pad weights and bias
            weight = weights[weights_max]*F.pad(
                    weight, (0, self.max_in_dim-weight.shape[-1], 0, self.max_out_dim - weight.shape[-2]), "constant", 0)
            bias = weights[weights_max]*F.pad(bias, (0, self.max_out_dim -
                            bias.shape[-1]), "constant", 0)
            out = F.linear(x, weight, bias)
        else:
            weights_mix = 0
            bias_mix = 0
            k = 0
            for i in range(len(self.input_dim_list)):
                for j in range(len(self.output_dim_list)):
                    if self.reverse:
                        weight, bias = self.sample_weights_and_bias(
                        self.output_dim_list[j]*self.input_dim_list[i], self.input_dim_list[i], self.linear_layer)
                    else:
                        weight, bias = self.sample_weights_and_bias(
                        self.input_dim_list[i], self.output_dim_list[j]*self.input_dim_list[i], self.linear_layer)
                    # pad weights and bias
                    # print(weight.shape)
                    weight = F.pad(
                    weight, (0, self.max_in_dim-weight.shape[-1], 0, self.max_out_dim - weight.shape[-2]), "constant", 0)
                    # print(weight.shape)
                    bias = F.pad(bias, (0, self.max_out_dim -
                             bias.shape[-1]), "constant", 0)
                    weights_mix += weights[k]*weight
                    bias_mix += weights[k]*bias
                    k = k+1
            out = F.linear(x, weights_mix, bias_mix)

        return out


class MixedLinearHead(nn.Module):
    def __init__(self, input_dim, max_embed_dim):
        super().__init__()
        self.input_dim = input_dim
        self.max_embed_dim = max_embed_dim

    def sample_weights_and_bias(self, linear_layer):
        weight = linear_layer.weight[:, :self.input_dim]
        bias = linear_layer.bias
        return weight, bias

    def forward(self, x, linear_layer):
        weight, bias = self.sample_weights_and_bias(linear_layer)
        out = F.linear(x[:, :, :self.input_dim], weight, bias)
        return out


class MixedLinearHeadV2(nn.Module):
    def __init__(self, input_dim_list, max_embed_dim, linear_layer):
        super().__init__()
        self.input_dim_list = input_dim_list
        self.linear_layer = linear_layer
        self.max_embed_dim = max_embed_dim

    def sample_weights_and_bias(self, input_dim, linear_layer):
        weight = linear_layer.weight[:, :input_dim]
        bias = linear_layer.bias
        return weight, bias

    def forward(self, x, weights, use_argmax=False):
        if use_argmax:
            weights = torch.tensor(weights)
            weights_max = torch.argmax(weights, dim=-1)
            weight, bias = self.sample_weights_and_bias(
                self.input_dim_list[weights_max], self.linear_layer)
            # pad weights and bias
            weight = F.pad(weight, (0, self.max_embed_dim -weight.shape[-1]), "constant", 0)
            out = F.linear(x, weight, bias)
        else:
            weights_mix = 0
            # print(x.shape)
            for i in range(len(self.input_dim_list)):
                weight, bias = self.sample_weights_and_bias(
                self.input_dim_list[i], self.linear_layer)
                # pad weights and bias
                # print("Heeey",weight.shape)
                weight = F.pad(weight, (0, self.max_embed_dim -
                           weight.shape[-1]), "constant", 0)
                # print("Heeey",weight.shape)
                # print(bias.shape)
                weights_mix += weights[i]*weight
            out = F.linear(x, weights_mix, bias)

        return out


'''# Test mixed linear layer
linear_layer = nn.Linear(10, 20)
mixed_linear = MixedLinear(5, 10, 20)
x = torch.randn(2, 2, 10)
out = mixed_linear(x, linear_layer)
print(out.shape)'''
