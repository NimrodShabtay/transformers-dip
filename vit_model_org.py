import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from torch import nn
from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchinfo import summary
import torchvision.datasets as datasets
from torch.utils.data import DataLoader


transform = Compose([
    ToTensor()
])

cifar_trainset = datasets.CIFAR10(root='./data',
                                  train=True,
                                  download=False,
                                  transform=transform)

cifar_testset = datasets.CIFAR10(root="./data",
                                   train=False,
                                   download=False,
                                   transform=transform)


BATCH_SIZE = 1
IMG_DIM = 32
PATCH_SIZE = 2
device = 'cuda' if torch.cuda.is_available() else 'cpu'


train_loader = DataLoader(cifar_trainset,
                          batch_size=BATCH_SIZE,
                          num_workers=1)

test_loader = DataLoader(cifar_testset,
                         batch_size=BATCH_SIZE,
                         num_workers=1)


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = PATCH_SIZE, emb_size: int = 768, img_size=IMG_DIM):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            # using a conv layer instead of a linear one -> performance gains
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.positions = nn.Parameter(torch.randn((img_size // patch_size) ** 2 + 1, emb_size))

    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        # prepend the cls token to the input
        x = torch.cat([cls_tokens, x], dim=1)
        # add position embedding
        x += self.positions

        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 768, num_heads: int = 8, dropout: float = 0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        # split keys, queries and values in num_heads
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        # sum up over the last axis
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy, dim=-1) / scaling
        att = self.att_drop(att)
        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size: int = 768,
                 drop_p: float = 0.,
                 forward_expansion: int = 4,
                 forward_drop_p: float = 0.,
                 ** kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads=8),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth: int = 12, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size: int = 768, n_classes: int = 1000):
        super().__init__(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes))


class ViT(nn.Sequential):
    def __init__(self,
                in_channels: int = 3,
                patch_size: int = PATCH_SIZE,
                emb_size: int = 768,
                img_size: int = IMG_DIM,
                depth: int = 6,
                n_classes: int = 10,
                **kwargs):
        super().__init__(
            PatchEmbedding(in_channels, patch_size, emb_size, img_size),
            TransformerEncoder(depth, emb_size=emb_size, **kwargs),
            ClassificationHead(emb_size, n_classes)
        )


if __name__ == '__main__':
    vit = ViT().to(device)
    summary(vit, input_size=(1, 3, 32, 32))

    LR = 0.001
    NUM_EPOCHS = 5
    WARMS_UP = 2

    optimizer = torch.optim.Adam(vit.parameters(),
                                 lr=LR)
    # scheduler = WarmupCosineSchedule(optimizer, warmup_steps=WARM_UP, t_total=NUM_EPOCHS)

    vit.zero_grad()
    torch.manual_seed(0)  # Added here for reproducibility (even between python 2 and 3)
    loss_criteria = loss_fct = torch.nn.CrossEntropyLoss()
    running_loss = 0.0
    epoch_loss = []
    global_step, best_acc = 0, 0
    vit.train()
    for epoch in range(NUM_EPOCHS):
        for n, batch in enumerate(train_loader):
            if n > 5:
                break
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            preds = vit(x)
            loss = loss_criteria(preds, y)
            running_loss += loss.item()
            loss.backward()
            # scheduler.step()
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            print('Running Loss(Batch #{}): {}'.format(n, running_loss))

        epoch_loss.append(running_loss / n)
        running_loss = 0.0
        print('Epoch Loss (Epoch #{}): {}'.format(epoch, epoch_loss[-1]))

    plt.plot(epoch_loss)
    plt.show()
