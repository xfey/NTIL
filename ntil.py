"""
Numerical Token Integrity Loss (NTIL)

This module implements the Numerical Token Integrity Loss (NTIL), a specialized loss function 
designed to improve numerical prediction accuracy in language models. NTIL addresses the limitations 
of traditional cross-entropy loss when dealing with numerical values by considering the semantic 
relationships between numbers.

Key features:
1. Token-level optimization: Replaces traditional CE loss with Earth Mover's Distance (EMD)
   to better capture the numerical relationships between digits.
2. Exponential Position-based Weighting: Applies higher weights to more significant digit positions
   to respect place-value number systems.
3. Sequence-level numerical optimization: Evaluates the overall numerical difference between
   predicted and actual sequences holistically.
4. Multi-component loss: Combines digit-wise EMD loss with absolute numerical difference and
   order of magnitude difference to capture various aspects of numerical similarity.

The loss function helps language models better understand and generate numerical content by
penalizing predictions based on their numerical distance rather than just token-level accuracy.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F


class NTIL(nn.Module):
    def __init__(self, 
                 tokenizer=None, 
                 batch_size=-1, 
                 ntil_lambda=0.3,
                 abs_alpha=0.2,
                 mag_alpha=0.2,
                 digit_alpha=1.0,
                 digit_exp=1.2,
                 ):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.batch_size = batch_size        # to avoid numbers leak between batches
        self.num_batches = 0                # temp saving
        self.num_vocab = 0                  # temp saving

        self.ntil_lambda = ntil_lambda                  # overall lambda
        self.abs_alpha = abs_alpha          # absolute numerical (relative)
        self.mag_alpha = mag_alpha          # order of magnitude
        self.digit_alpha = digit_alpha      # emd loss (w_dist)
        self.digit_exp = digit_exp          # exp increase for higher digit position

        self.device = None
        
        # 预计算常用的常量张量
        self.zero_tensor = torch.tensor(0.)
        self.one_tensor = torch.tensor(1.)
        self.digit_arange = torch.arange(10)  # 数字范围0-9
        self.digit_points = self.digit_arange.view(-1, 1).float()  # 用于计算距离的点
        
        # 预计算距离矩阵
        self.digit_distances = torch.cdist(self.digit_points, self.digit_points, p=1)

        self.parse_from_tokenizer = (tokenizer is not None)
        if self.parse_from_tokenizer:
            # Get token indices for each digit
            digit_cols = []
            self.digit_token_map = {}
            for digit in [str(i) for i in range(10)]:
                token_id = tokenizer.convert_tokens_to_ids(digit)
                assert isinstance(token_id, int) or (len(token_id) == 1), \
                    f"digital token contains more than one value, {token_id}"
                digit_cols.append(token_id)
                self.digit_token_map[token_id] = digit
            # 将数字列表转换为张量以提高效率
            self.digit_columns = torch.tensor(digit_cols)
            # 预先创建数字token集合，用于加速查找
            self.digit_token_set = set(digit_cols)
        else:
            print('No tokenizer: naive matching for the whole, assert related tokens in order')
            self.digit_columns = None
            self.digit_token_set = None

    def _ensure_device(self, device):
        """确保所有缓存的张量在正确的设备上"""
        if self.device != device:
            self.device = device
            self.zero_tensor = self.zero_tensor.to(device)
            self.one_tensor = self.one_tensor.to(device)
            self.digit_arange = self.digit_arange.to(device)
            self.digit_points = self.digit_points.to(device)
            self.digit_distances = self.digit_distances.to(device)
            if self.digit_columns is not None:
                self.digit_columns = self.digit_columns.to(device)

    def gumbel_softmax(self, logits, hard=True, tau=0.1, dim=-1, noise_scale=0.01):
        gumbels = -torch.empty_like(logits).exponential_().log() * noise_scale  # ~Gumbel(0,1)
        gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
        y_soft = gumbels.softmax(dim)

        if hard:
            # Straight through.
            index = y_soft.max(dim, keepdim=True)[1]
            y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
            ret = y_hard - y_soft.detach() + y_soft
        else:
            # Reparametrization trick.
            ret = y_soft
        return ret

    def differentiable_number_construction(self, logits):
        # logits shape: [batch_size, num_positions, num_digits]
        soft_digits = self.gumbel_softmax(logits)
        # 使用缓存的张量，并根据实际需要创建位置权重
        positions = torch.tensor([10.0 ** i for i in range(logits.shape[0]-1, -1, -1)], device=logits.device)
        # 使用缓存的数字范围张量，适配数据类型
        arange_tensor = self.digit_arange
        if soft_digits.dtype == torch.bfloat16:
            arange_tensor = arange_tensor.bfloat16()
        elif soft_digits.dtype == torch.float:
            arange_tensor = arange_tensor.float()
        values = torch.matmul(soft_digits, arange_tensor)
        # Combine positions
        final_numbers = torch.sum(values * positions, dim=-1)
        return final_numbers

    def number_construction_for_label(self, label:list):
        # 创建位置权重 (powers of 10)
        positions = torch.tensor([10.0 ** i for i in range(len(label)-1, -1, -1)], device=self.device)
        # 直接使用设备上的张量
        final_numbers = torch.sum(torch.tensor(label, device=self.device) * positions, dim=-1)
        return final_numbers

    def get_abs_values(self, x_i, y_i):
        ## no need to pass softmax for x
        abs_x_i = self.differentiable_number_construction(x_i)
        abs_y_i = self.number_construction_for_label(y_i)
        return abs_x_i, abs_y_i

    def split_data_sequences(self, x_batch, y_batch):
        # 使用预先创建的digit_token_set加速查找
        mask_batch = torch.tensor([token.item() in self.digit_token_set for token in y_batch], device=y_batch.device)
        split_indices = torch.where(mask_batch == False)[0].tolist()
        # Split data based on continuous True values in the mask
        start = 0
        x_splits, y_splits = [], []
        for idx in split_indices + [len(mask_batch)]:
            if start < idx:
                x_splits.append(x_batch[start:idx, self.digit_columns])     # to [N, 0-1]
                y_batch_cut = y_batch[start:idx]
                y_batch_real_digit = [int(self.digit_token_map[int(t)]) for t in y_batch_cut]
                y_splits.append(y_batch_real_digit)
            start = idx + 1  # Move start to the next segment after False
        return x_splits, y_splits

    def compute_abs_numerical_loss(self, x_batch, y_batch):
        # 整个数字绝对值loss = abs(a-b)/max(a,b)
        x_splits, y_splits = self.split_data_sequences(x_batch, y_batch)
        all_abs_losses = []
        for x_batch_i, y_batch_i in zip(x_splits, y_splits):
            abs_x_i, abs_y_i = self.get_abs_values(x_batch_i, y_batch_i)
            abs_diff = (torch.abs(abs_x_i - abs_y_i)) / (torch.maximum(abs_x_i, abs_y_i) + self.one_tensor)
            
            ## set to zeros if x and y are identical
            abs_diff = torch.where(abs_x_i == abs_y_i, self.zero_tensor, abs_diff)
            all_abs_losses.append(abs_diff)
        if len(all_abs_losses):
            abs_diff = torch.stack(all_abs_losses).mean()
        else:
            abs_diff = self.zero_tensor
        return abs_diff
        
    def compute_order_magnitude_loss(self, x_batch, y_batch):
        # 整个数字数量级loss = max(a,b)/min(a,b)
        x_splits, y_splits = self.split_data_sequences(x_batch, y_batch)
        all_mag_losses = []
        for x_batch_i, y_batch_i in zip(x_splits, y_splits):
            abs_x_i, abs_y_i = self.get_abs_values(x_batch_i, y_batch_i)
            mag_diff = torch.log((torch.maximum(abs_x_i, abs_y_i) + self.one_tensor) / (torch.minimum(abs_x_i, abs_y_i) + self.one_tensor))
            all_mag_losses.append(mag_diff)
        if len(all_mag_losses):
            mag_diff = torch.stack(all_mag_losses).mean()
        else:
            mag_diff = self.zero_tensor
        return mag_diff
    
    def compute_digit_wdist_loss(self, x_batch, y_batch):
        # 单个字符emd loss
        mask_batch = [token.item() in self.digit_token_set for token in y_batch]
        weights = torch.zeros(len(mask_batch), dtype=torch.int, device=self.device)
        # Accumulate weights from the end to the start
        current_weight = 0
        for i in reversed(range(len(mask_batch))):
            if mask_batch[i]:
                weights[i] = current_weight
                current_weight += 1
            else:
                current_weight = 0

        if any(mask_batch):
            weights_filtered = weights[mask_batch]
            exp_weights = self.digit_exp ** weights_filtered
            y_filtered = y_batch[mask_batch]  # [num_digit]
            # get real digit
            y_real_digit = [int(self.digit_token_map[int(t)]) for t in y_filtered]
            x_filtered = x_batch[mask_batch]  # [num_digit, vocab_size]
            x_digit = x_filtered[:, self.digit_columns]  # [num_digit, 10] in order of 0-9
            x_logits = self.softmax(x_digit)  # [num_digit, 10]
            # 使用预计算的距离矩阵
            sample_distances = self.digit_distances[y_real_digit]
            emd_values = torch.sum(sample_distances * x_logits, dim=1)
            # add exponential weights
            exp_emd_values = torch.mul(emd_values, exp_weights)
            emd_diff = torch.mean(exp_emd_values)
        else:
            # no digit to calculate, avoid nan
            emd_diff = self.zero_tensor
        
        return emd_diff

    def compute_emd_with_tokenizer(self, x, y):
        if self.batch_size == -1:
            self.batch_size = x.shape[0]
        # Reshape inputs back to batched form
        num_total, num_vocab = x.shape[0], x.shape[1]
        num_batches = num_total // self.batch_size
        if self.num_batches == 0 and self.num_vocab == 0:
            self.num_batches = num_batches
            self.num_vocab = num_vocab
        
        if self.batch_size * num_batches == x.shape[0]:
            x = x.view(self.batch_size, num_batches, -1)  # [num_batches, batch_size, vocab_size]
            y = y.view(self.batch_size, num_batches)      # [num_batches, batch_size]
        else:
            try:
                x = x.view(-1, self.num_batches, self.num_vocab)
                y = y.view(-1, self.num_batches)
            except:
                print(f"Error when reshaping tensor: x.shape={x.shape} batch_size={self.batch_size} \
                        num_batch={self.num_batches} len_vocab={self.num_vocab}")
                return self.zero_tensor
        
        batch_losses = []
        try:
            for batch_idx in range(self.batch_size):
                x_batch = x[batch_idx]  # [batch_size, vocab_size]
                y_batch = y[batch_idx]  # [batch_size]
                # Process each batch
                emd_diff = self.compute_digit_wdist_loss(x_batch, y_batch)
                abs_diff = self.compute_abs_numerical_loss(x_batch, y_batch)
                mag_diff = self.compute_order_magnitude_loss(x_batch, y_batch)

                # mixed loss
                mix_diff = self.digit_alpha * emd_diff + \
                        self.abs_alpha * abs_diff + \
                        self.mag_alpha * mag_diff
            
                batch_losses.append(mix_diff)
        except Exception as e:
            print("Error when calculating loss:", e)
            return self.zero_tensor
        # Calculate mean EMD across all batches
        final_loss = torch.stack(batch_losses).mean()
        return final_loss

    def compute_emd_directly(self, x, y):
        # currently used in ARTrack, Pix2Seq only.
        # calculating emd loss only

        x_logits = self.softmax(x)
        classes = x_logits.shape[1]

        # 使用预计算的点和距离，根据需要调整尺寸
        if classes == 10:
            # 数字类别为10时，直接使用预计算值
            distances = self.digit_distances
        else:
            # 非数字时，需要重新计算
            points = torch.arange(classes).view(-1, 1).float().to(x_logits.device)
            distances = torch.cdist(points, points, p=1)
            
        sample_distances = distances[y]
        emd_values = torch.sum(sample_distances * x_logits, dim=1)
        emd = torch.mean(emd_values)
        return emd

    def forward(self, x, y, mle_loss_value=None):
        # 在forward开始时确保所有缓存张量在正确的设备上
        self._ensure_device(x.device)

        if self.ntil_lambda == 0.0:
            return mle_loss_value

        if self.parse_from_tokenizer:
            emd = self.compute_emd_with_tokenizer(x,y)
        else:
            emd = self.compute_emd_directly(x,y)

        if mle_loss_value is not None and self.ntil_lambda != 0.0:
            mle = mle_loss_value
            norm_emd = (mle/(emd+1e-10)).detach() * self.ntil_lambda * emd
            return norm_emd + mle
        else:
            return emd


if __name__ == "__main__":
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("./models/llava-1.5-7b-hf")

    BS, L, vocab = 4, 5, len(tokenizer)
    x = torch.randn(BS * L, vocab, requires_grad=True)
    # y = torch.randint(0, vocab, (BS * L,))
    y_label = ['3.015', '5123 ', '89()2', '[9_4]']

    digits = [29900, 29896, 29906, 29941, 29946, 29945, 29953, 29955, 29947, 29929]

    y_tokens = []
    for label in y_label:
        tokens = []
        for char in label:
            tokens.append(tokenizer.convert_tokens_to_ids(char))
        y_tokens.extend(tokens)
    y = torch.tensor(y_tokens)

    # print(y)
    # tensor([29941, 29889, 29900, 29896, 29945, 29945, 29896, 29906, 29941,     0,
    #         29947, 29929, 29898, 29897, 29906, 29961, 29929, 29918, 29946, 29962])
    # print([t in digits for t in y_tokens])
    # [True, False, True, True, True, True, True, True, True, False, True, True, False, False, True, False, True, False, True, False]

    # Initialize loss function
    digit_loss = NTIL(
        tokenizer=tokenizer,
        batch_size=BS,
        ntil_lambda=0.3,
        abs_alpha=0.0,
        mag_alpha=0.0,
        digit_alpha=1.0,
        digit_exp=1.2,
    )

    # Calculate loss
    loss = digit_loss(x, y)

    loss.backward()
    print(x.grad.abs().sum(dim=-1))
