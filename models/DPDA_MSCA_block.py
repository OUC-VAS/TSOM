import copy
from typing import Optional
import math
import torch
import torch.nn.functional as F
from torch import nn, Tensor

from .attention_layer import GaussianMultiheadAttention
from torch.distributions import Normal
from .containers import Fusion


# Seq_transformer model
class Seq_Transformer(nn.Module):

    def __init__(self, config, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout1=0.1, dropout2=0.3,
                 activation="gelu", normalize_before=False,
                 return_intermediate_dec=False, lambda1=0.2, lambda2=0.01, fla_dim=64, alpha=0.005, dim_fac=108):
        super().__init__()

        encoder_layer = Seq_TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout1, activation, normalize_before, dim_fac)

        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None

        self.encoder = Seq_TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        self.embeddings = DecoderEmbeddings(config)
        decoder_layers = []
        for layer_index in range(num_decoder_layers):
            decoder_layer = Seq_TransformerDecoderLayer(layer_index, d_model, nhead, dim_feedforward, dropout2, activation,
                                                    normalize_before, lambda1, lambda2, fla_dim, alpha)
            decoder_layers.append(decoder_layer)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = Seq_TransformerDecoder(decoder_layers, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, pos_embed, target, tgt_mask, h_w, image):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape

        grid_y, grid_x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))  # 8-8
        grid = torch.stack((grid_x, grid_y), dim=2).float().to(src.device)

        grid = grid.reshape(-1, 2).unsqueeze(1).repeat(1, bs * self.nhead, 1)  # 64*128*2

        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)

        tgt = self.embeddings(target).permute(1, 0, 2)
        query_embed = self.embeddings.position_embeddings.weight.unsqueeze(1)
        query_embed = query_embed.repeat(1, bs, 1)

        memory, memory_cda = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)

        hs, loss_vae = self.decoder(grid, image, tgt, memory, memory_cda, memory_key_padding_mask=mask,
                                    tgt_key_padding_mask=tgt_mask, pos=pos_embed, query_pos=query_embed,
                                    tgt_mask=generate_square_subsequent_mask(len(tgt)).to(tgt.device))

        return hs, loss_vae


# Diversiform Pixel Difference conv block
class Conv2d_DPDA(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1,
                 groups=1, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):

        #central difference conv angular, and radial
        out_normal = self.conv(x)
        # pdb.set_trace()
        [C_out, C_in, kernel_size, kernel_size] = self.conv.weight.shape
        kernel_diff = self.conv.weight.sum(2).sum(2)
        kernel_diff = kernel_diff[:, :, None, None]
        out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias,
                            stride=self.conv.stride, padding=0, dilation=1, groups=self.conv.groups)
        out_cpdc = out_normal - out_diff

        #angular difference conv
        shape = self.conv.weight.shape
        weights = self.conv.weight.view(shape[0], shape[1], -1)
        weights_conv = (weights - weights[:, :, [3, 0, 1, 6, 4, 2, 7, 8, 5]]).view(shape)
        out_apdc = F.conv2d(input=x, weight=weights_conv, bias=self.conv.bias, stride=self.conv.stride,
                            padding=self.conv.padding, dilation=1, groups=self.conv.groups)

        # radial difference conv
        buffer = torch.cuda.FloatTensor(shape[0], shape[1], 5 * 5).fill_(0)
        weights_two = self.conv.weight.view(shape[0], shape[1], -1)
        buffer[:, :, [0, 2, 4, 10, 14, 20, 22, 24]] = weights_two[:, :, 1:]
        buffer[:, :, [6, 7, 8, 11, 13, 16, 17, 18]] = -weights_two[:, :, 1:]
        buffer[:, :, 12] = 0
        buffer = buffer.view(shape[0], shape[1], 5, 5)
        out_rpdc = F.conv2d(input=x, weight=buffer, bias=self.conv.bias, stride=self.conv.stride,
                            padding=2, groups=self.conv.groups, dilation=1)

        return out_cpdc, out_apdc, out_rpdc


# Seq_transforemr Encoder
class Seq_TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output1 = src
        output2 = src

        for layer in self.layers:
            output1, output2 = layer(output1, output2, src_mask=mask,
                                     src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output1 = self.norm(output1)
            output2 = self.norm(output2)

        return output1, output2



class Seq_TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, dim_fac=108, basic_conv=Conv2d_DPDA):
        super().__init__()
        self.attn1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.attn2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # rgb refers to  the vanilla branch and cda(central difference attn) refers to the texture_branch(DPDA)
        self.norm1_rgb = nn.LayerNorm(d_model)
        self.norm1_cda = nn.LayerNorm(d_model)

        self.linear1_rgb = nn.Linear(d_model, dim_feedforward)
        self.dropout_rgb = nn.Dropout(dropout)
        self.linear2_rgb = nn.Linear(dim_feedforward, d_model)

        self.norm2_rgb = nn.LayerNorm(d_model)
        self.dropout1_rgb = nn.Dropout(dropout)
        self.dropout2_rgb = nn.Dropout(dropout)

        self.activation_rgb = _get_activation_fn(activation)

        self.linear1_cda = nn.Linear(d_model, dim_feedforward)
        self.dropout_cda = nn.Dropout(dropout)
        self.linear2_cda = nn.Linear(dim_feedforward, d_model)

        self.norm2_cda = nn.LayerNorm(d_model)
        self.dropout1_cda = nn.Dropout(dropout)
        self.dropout2_cda = nn.Dropout(dropout)

        self.activation_cda = _get_activation_fn(activation)

        #q: vanilla conv  kv: Diversiform Pixel Difference conv
        self.conv_q = nn.Conv2d(d_model, d_model, kernel_size=3, stride=1, padding=1)
        self.conv_k = basic_conv(d_model, d_model, kernel_size=3, stride=1, padding=1)
        self.conv_v = basic_conv(d_model, d_model, kernel_size=3, stride=1, padding=1)

        self.normalize_before = normalize_before

        modules = []
        hidden_dims = [128, 64, 32]
        input_channel = 256

        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(input_channel, out_channels=h_dim,
                              kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(),

                )
            )
            input_channel = h_dim
        modules.append(nn.Conv2d(32, 3, kernel_size=3, stride=1))
        self.encoder = nn.Sequential(*modules)
        self.fac = nn.Linear(in_features=dim_fac, out_features=3)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_pre(self, src_rgb, src_cda,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2_rgb = self.norm1_rgb(src_rgb)
        q1 = k1 = v1 = self.with_pos_embed(src2_rgb, pos)
        s, bs, c = src_cda.shape
        h = w = int(math.sqrt(s))
        src2_cda = self.norm1_cda(src_cda)

        src3_cda = src2_cda.contiguous().view(bs, c, h, w)
        pos_cda = pos.contiguous().view(bs, c, h, w)

        src_cda2 = self.with_pos_embed(src3_cda, pos_cda)  # shape(32,256,8,8)

        #get the vanilla transformer feature
        rgb_att = self.attn1(q1, k1, v1, attn_mask=src_mask,
                             key_padding_mask=src_key_padding_mask)[0]

        rgb_ff = src_rgb + self.dropout1_rgb(rgb_att)
        rgb_ff2 = self.norm2_rgb(rgb_ff)
        rgb_ff2 = self.linear2_rgb(self.dropout_rgb(self.activation_rgb(self.linear1_rgb(rgb_ff2))))
        rgb = rgb_ff + self.dropout2_rgb(rgb_ff2)

        #get the texture feature using the DPDA
        fac = self.encoder(src_cda2).flatten(1)
        fac = self.fac(fac)
        fac = F.softmax(fac).unsqueeze(2).unsqueeze(3)
        q = self.conv_q(src_cda2)
        k_cp, k_ap, k_rp = self.conv_k(src_cda2)
        k = fac[:, 0:1, :, :] * k_cp + fac[:, 1:2, :, :] * k_ap + fac[:, 2:3, :, :] * k_rp
        v_cp, v_ap, v_rp = self.conv_v(src_cda2)
        v = fac[:, 0:1, :, :] * v_cp + fac[:, 1:2, :, :] * v_ap + fac[:, 2:3, :, :] * v_rp

        q = q.flatten(2).permute(2, 0, 1)
        k = k.flatten(2).permute(2, 0, 1)
        v = v.flatten(2).permute(2, 0, 1)
        cda_att = self.attn2(q, k, v, attn_mask=src_mask,
                             key_padding_mask=src_key_padding_mask)[0]

        cda_ff = src_cda + self.dropout1_cda(cda_att)
        cda_ff2 = self.norm2_cda(cda_ff)
        cda_ff2 = self.linear2_cda(self.dropout_cda(self.activation_cda(self.linear1_cda(cda_ff2))))
        cda = cda_ff + self.dropout2_cda(cda_ff2)

        return rgb, cda

    def forward(self, src_rgb, src_cda,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src_rgb, src_cda, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src_rgb, src_cda, src_mask, src_key_padding_mask, pos)

#Seq_transformer Decoder
class Seq_TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = nn.ModuleList(decoder_layer)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, grid, image, tgt, memory, memory_cda,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                ):
        output = tgt

        intermediate = []
        mu_pre = None   # mu(mean) and log_var_pre(Variance) in Shape-guided Gaussian Mapping
        log_var_pre = None
        result_pre = None
        loss_vae = None

        for layer in self.layers:
            output, loss_vae, mu_pre, log_var_pre, result_pre = layer(
                grid, image, output, memory, memory_cda, tgt_mask=tgt_mask,
                memory_mask=memory_mask, tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos, query_pos=query_pos, mu_pre=mu_pre,
                log_var_pre=log_var_pre, result_pre=result_pre, loss_vae=loss_vae
            )
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output, loss_vae


class Seq_TransformerDecoderLayer(nn.Module):

    def __init__(self, layer_index, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, lambda1=0.2, lambda2=0.01, fla_dim=64, alpha=0.005):
        super().__init__()

        self.self_attn1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.multihead_attn1 = GaussianMultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn2 = GaussianMultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn3 = GaussianMultiheadAttention(d_model, nhead, dropout=dropout)

        # a mlp layer that fuse the  fine-grained spatial traces and  Coarse-grained features
        self.fusion = Fusion()

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.norm5 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)
        self.dropout5 = nn.Dropout(dropout)
        self.dropout6 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.nhead = nhead
        self.layer_index = layer_index
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.fla_dim = fla_dim  #a Dimension scaling factor
        self.alpha = alpha    # attention balance factor

        self.point1 = nn.Linear(256 * fla_dim, 48)
        self.point2 = nn.Linear(256 * fla_dim, 48)

        if layer_index == 0:
            in_channels = 3
            modules1 = []
            hidden_dims = [32, 64, 128, 256, 256]
            # build encoder for gaussian
            for h_dim in hidden_dims:
                modules1.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, out_channels=h_dim,
                                  kernel_size=3, stride=2, padding=1),
                        nn.BatchNorm2d(h_dim),
                        nn.LeakyReLU()
                    )
                )
                in_channels = h_dim
            self.encoder = nn.Sequential(*modules1)
            self.fc_mu = nn.Linear(hidden_dims[-1] * fla_dim, 12)
            self.fc_var = nn.Linear(hidden_dims[-1] * fla_dim, 12)

            # build decoder
            modules2 = []
            self.decoder_input = nn.Linear(12, hidden_dims[-1] * fla_dim)
            hidden_dims.reverse()
            for i in range(len(hidden_dims) - 1):
                modules2.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(
                            hidden_dims[i],
                            hidden_dims[i + 1],
                            kernel_size=3,
                            stride=2,
                            padding=1,
                            output_padding=1
                        ),
                        nn.BatchNorm2d(hidden_dims[i + 1]),
                        nn.LeakyReLU()
                    )
                )
            self.decoder = nn.Sequential(*modules2)
            self.final_layer = nn.Sequential(
                nn.ConvTranspose2d(hidden_dims[-1],
                                   hidden_dims[-1],
                                   kernel_size=3,
                                   stride=2,
                                   padding=1,
                                   output_padding=1
                                   ),
                nn.BatchNorm2d(hidden_dims[-1]),
                nn.LeakyReLU(),
                nn.Conv2d(hidden_dims[-1], out_channels=3,
                          kernel_size=3, padding=1),
                nn.Tanh()
            )

    # vae encoder
    def encode(self, input):
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)  # w维度？
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return [mu, log_var, result]

    #vae decoder
    def decode(self, z, bs):
        result = self.decoder_input(z)
        size = int(math.sqrt(self.fla_dim))
        result = result.view(bs, -1, size, size)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu


    #Calculate reconstruction loss and kld loss
    def loss_function(self, recons, input, mu, log_var):
        recon_loss = F.mse_loss(recons, input)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        loss_vae = recon_loss * self.lambda1 + kld_loss * self.lambda2
        return {'loss': loss_vae, 'Reconstruction_loss': recon_loss.detach(), 'KLD': kld_loss.detach()}

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos


    def forward_pre(self, grid, image, tgt, memory, memory_cda,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None,
                    mu_pre: Optional[Tensor] = None,
                    log_var_pre: Optional[Tensor] = None,
                    result_pre: Optional[Tensor] = None,
                    loss_vae: Optional[Tensor] = None
                    ):

        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn1(q, k, value=tgt2, attn_mask=tgt_mask,
                               key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)

        tgt_len = tgt.shape[0]
        s, bs, dim = memory.size()
        h = w = int(math.sqrt(s))

        # Calculate reconstruction loss and get the shape_guided gaussian map, only the first layer need to calculate the mean and variance
        if self.layer_index == 0:
            mu, log_var, result = self.encode(image)
            z = self.reparameterize(mu, log_var)
            output = self.decode(z, bs)
            mu2, log_var2 = mu.contiguous().view(6, bs, 2), log_var.contiguous().view(6, bs, 2)
            mu2, log_var2 = mu2.repeat(1, 1, self.nhead).sigmoid() * 4, log_var2.repeat(1, 1, self.nhead).exp()

            mu2_offset, log_var2_offset = self.point1(result).view(6, bs, 8).sigmoid(), self.point2(result).view(6, bs,
                                                                                                                 8).sigmoid()
            mu3, log_var3 = mu2 + mu2_offset, log_var2 + log_var2_offset
            mu3, log_var3 = mu3.view(tgt_len, -1, 2), log_var3.view(tgt_len, -1, 2)

            gau_cen = Normal(mu3, log_var3)
            grid = grid.unsqueeze(0).expand(6, -1, -1, -1).permute(1, 0, 2, 3)
            prob_cen = gau_cen.log_prob(grid).permute(1, 0, 2, 3)
            gaussian = prob_cen.sum(3)
            loss_vae = self.loss_function(output, image, mu, log_var)
        else:
            mu2, log_var2, result = mu_pre, log_var_pre, result_pre
            mu2_offset, log_var2_offset = self.point1(result).view(6, bs, 8).sigmoid(), self.point2(result).view(6, bs,
                                                                                                                 8).sigmoid()
            mu2, log_var2 = mu2 + mu2_offset, log_var2 + log_var2_offset
            mu2, log_var2 = mu2.view(tgt_len, -1, 2), log_var2.view(tgt_len, -1, 2)

            gau_cen = Normal(mu2, log_var2)
            grid = grid.unsqueeze(0).expand(6, -1, -1, -1).permute(1, 0, 2, 3)
            prob_cen = gau_cen.log_prob(grid).permute(1, 0, 2, 3)
            gaussian = prob_cen.sum(3)
            loss_vae = loss_vae

        feature = torch.concat((memory, memory_cda), dim=-1)
        feature = self.fusion(feature)

        #get the a_m
        tgt_one = self.multihead_attn1(query=self.with_pos_embed(tgt, query_pos),
                                       key=self.with_pos_embed(feature, pos),
                                       value=feature, attn_mask=memory_mask,
                                       key_padding_mask=memory_key_padding_mask,
                                       gaussian=[gaussian], alpha=self.alpha)[0]
        tgt_one = self.norm3(self.dropout5(tgt_one) + tgt)

        #get the a_c
        tgt_two = self.multihead_attn2(query=self.with_pos_embed(tgt, query_pos),
                                       key=self.with_pos_embed(memory, pos),
                                       value=memory, attn_mask=memory_mask,
                                       key_padding_mask=memory_key_padding_mask,
                                       gaussian=[gaussian], alpha=self.alpha)[0]
        tgt_two = self.norm4(self.dropout2(tgt_two) + tgt)

        #get the a_f
        tgt_three = self.multihead_attn3(query=self.with_pos_embed(tgt, query_pos),
                                         key=self.with_pos_embed(memory_cda, pos),
                                         value=memory_cda, attn_mask=memory_mask,
                                         key_padding_mask=memory_key_padding_mask,
                                         gaussian=[gaussian], alpha=self.alpha)[0]
        tgt_three = self.norm5(self.dropout6(tgt_three) + tgt)
        tgt = tgt_one + tgt_two + tgt_three
        tgt2 = self.linear2(self.activation(self.linear1(tgt)))
        tgt = tgt + self.dropout4(tgt2)

        if self.layer_index == 0:
            return tgt, loss_vae, mu2, log_var2, result
        else:
            return tgt, loss_vae, None, None, None

    def forward(self, grid, image, tgt, memory, memory_cda,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                mu_pre: Optional[Tensor] = None,
                log_var_pre: Optional[Tensor] = None,
                result_pre: Optional[Tensor] = None,
                loss_vae: Optional[Tensor] = None, ):
        if self.normalize_before:
            return self.forward_pre(grid, image, tgt, memory, memory_cda, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos,
                                    mu_pre, log_var_pre, result_pre, loss_vae
                                    )
        return self.forward_post(grid, tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos,
                                 mu_pre, log_var_pre, result_pre, loss_vae)



def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class DecoderEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_dim, padding_idx=config.PAD_token_id)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_dim
        )

        self.LayerNorm = torch.nn.LayerNorm(
            config.hidden_dim, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout2)

    def forward(self, x):
        input_shape = x.size()
        seq_length = input_shape[1]
        device = x.device

        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(input_shape)

        input_embeds = self.word_embeddings(x)
        position_embeds = self.position_embeddings(position_ids)

        embeddings = input_embeds + position_embeds
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


def generate_square_subsequent_mask(sz):
    r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float(
        '-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def build_transformer(config):
    return Seq_Transformer(
        config,
        d_model=config.hidden_dim,
        dropout1=config.dropout1,
        dropout2=config.dropout2,
        nhead=config.nheads,
        dim_feedforward=config.dim_feedforward,
        num_encoder_layers=config.enc_layers,
        num_decoder_layers=config.dec_layers,
        normalize_before=config.pre_norm,
        return_intermediate_dec=False,
        lambda1=config.lambda1,
        lambda2=config.lambda2,
        alpha=config.alpha,
        dim_fac=config.dim_fac,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
