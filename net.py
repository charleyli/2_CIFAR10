import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


def testNet(workNet):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 使用GPU或者CPU
    model = workNet.to(DEVICE)

    x = torch.randn(1,3,32,32).to(DEVICE)
    y = model(x)
    print(y.size())

    summary(model,(3,32,32))


class CIFAR10Net(nn.Module):
    def __init__(self):
        super(CIFAR10Net, self).__init__()
        self.conv1 = nn.Conv2d(3,6,5)   # 输入3，输出6，卷积核5*5
        self.pool1 = nn.MaxPool2d(2,2) # kernel_size:2 stride:2
        self.conv2 = nn.Conv2d(6,16,5)  # 输入6，输出16，卷积核5*5
        self.fc1 = nn.Linear(16*5*5,120)  # 全连接层，输入16*5*5，输出120
        self.fc2 = nn.Linear(120,84)   # 全连接层，输入120，输出84
        self.fc3 = nn.Linear(84,10)   # 全连接层，输入84，输出10

    def forward(self,x):
        input_size = x.size(0) # [batch_size, 3, 32, 32]

        # 输出高度 = (输入高度 - 卷积核大小 + 2*填充) / 步幅 + 1 = (32 - 5 + 0) / 1 + 1 = 28
        x = self.conv1(x) # [batch_size, 3, 32, 32] --> [batch_size, 6, 28, 28] (32 - 5 + 2*0) / 1 + 1 = 28
        x = F.relu(x) # [batch_size, 6, 28, 28]
        
        # 输出尺寸 = floor( (输入尺寸 + 2*padding - kernel_size) / stride ) + 1
        x = self.pool1(x) # [batch_size, 6, 28, 28] --> [batch_size, 6, 14, 14] (28 + 2*0 - 2) / 2 + 1 = 14

        x= self.conv2(x) # [batch_size, 6, 14, 14] --> [batch_size, 16, 10, 10] (14 - 5 + 2*0) / 1 + 1 = 10
        x = F.relu(x) # [batch_size, 16, 10, 10]

        x = self.pool1(x) # [batch_size, 16, 10, 10] --> [batch_size, 16, 5, 5] (10 + 2*0 - 2) / 2 + 1 = 5

        x = x.view(input_size,-1) # [batch_size, 16, 5, 5] --> [batch_size, 16*5*5]

        x = self.fc1(x) # [batch_size, 16*5*5] --> [batch_size, 120]
        x = F.relu(x) # 保持输入大小不变 [batch_size, 120]

        x = self.fc2(x) # [batch_size, 120] --> [batch_size, 84]
        x = F.relu(x) # 保持输入大小不变 [batch_size, 84]

        x = self.fc3(x) # [batch_size, 84] --> [batch_size, 10]

        # output = F.log_softmax(x, dim=1) # [batch_size, 10] --> [batch_size, 10]
        return x
    

class CNNNet(nn.Module):
    def __init__(self):
        super(CNNNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32) #32是通道数，不改变维度
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64) #64是通道数，不改变维度
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128) #128是通道数，不改变维度
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(128) #128是通道数，不改变维度
        self.fc1 = nn.Linear(128*4*4, 256)
        self.dropout = nn.Dropout(0.5) # 每个元素有50%的概率变成0,不改变维度
        self.fc2 = nn.Linear(256, 10)
    def forward(self,x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x





class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])



class PatchEmbedding(nn.Module):
    """
    将输入图像分成 patch 并映射到 embedding 空间
    """
    def __init__(self, in_channels=3, patch_size=4, emb_size=128, img_size=32):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        # x: [B, C, H, W] -> [B, emb_size, H/patch, W/patch]
        x = self.proj(x)
        x = x.flatten(2)          # [B, emb_size, N_patches]
        x = x.transpose(1, 2)     # [B, N_patches, emb_size]
        return x

class Attention(nn.Module):
    """
    多头自注意力机制
    """
    def __init__(self, emb_size=128, num_heads=8, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.emb_size = emb_size
        assert emb_size % num_heads == 0, "Embedding size must be divisible by number of heads"
        self.head_dim = emb_size // num_heads

        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.fc_out = nn.Linear(emb_size, emb_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x)  # [B, N, 3*emb_size]
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # 3, B, heads, N, head_dim
        q, k, v = qkv[0], qkv[1], qkv[2]  # 每个都是 [B, heads, N, head_dim]

        # 计算注意力
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B, heads, N, N]
        attn = F.softmax(attn, dim=-1)
        out = attn @ v  # [B, heads, N, head_dim]
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.fc_out(out)
        out = self.dropout(out)
        return out

class TransformerEncoderBlock(nn.Module):
    """
    Transformer Encoder Block: LayerNorm + Multi-head Attention + MLP + Residual
    """
    def __init__(self, emb_size=128, num_heads=8, mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_size)
        self.attn = Attention(emb_size, num_heads, dropout)
        self.norm2 = nn.LayerNorm(emb_size)
        self.mlp = nn.Sequential(
            nn.Linear(emb_size, emb_size*mlp_ratio),
            nn.GELU(),
            nn.Linear(emb_size*mlp_ratio, emb_size),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class ViT(nn.Module):
    """
    简单的 Vision Transformer
    """
    def __init__(self, img_size=32, patch_size=4, in_channels=3, num_classes=10,
                 emb_size=128, depth=6, num_heads=8, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels, patch_size, emb_size, img_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_size))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.n_patches, emb_size))
        self.dropout = nn.Dropout(dropout)

        # Transformer encoder
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(emb_size, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(emb_size)
        self.head = nn.Linear(emb_size, num_classes)
    
    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)  # [B, N_patches, emb_size]
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, emb_size]
        x = torch.cat((cls_tokens, x), dim=1)          # [B, 1+N, emb_size]
        x = x + self.pos_embed
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        cls_token_final = x[:, 0]  # 取 cls token
        out = self.head(cls_token_final)
        return out



if __name__ == "__main__":
    # model = ResNet18()
    # model = ViT(img_size=32, patch_size=4, emb_size=128, depth=6)
    # testNet(model)
    exit
