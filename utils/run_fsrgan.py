import torch
from torchvision.models.vgg import VGG19_Weights, vgg19

# VGG19 Class: Modified VGG19 implementation for feature extraction
# VGG19 클래스: 특징 추출을 위해 수정된 VGG19 구현
class VGG19(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Initialize pretrained VGG19 model and freeze its parameters
        # 사전 훈련된 VGG19 모델을 초기화하고 파라미터를 고정
        self.vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[:34]
        for param in self.vgg.parameters():
            param.requires_grad = False
            
        # Register normalization parameters
        # 정규화 파라미터 등록
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406], requires_grad=False).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225], requires_grad=False).view(1, 3, 1, 1),
        )

    # Forward pass: normalize input and process through VGG
    # 순전파: 입력을 정규화하고 VGG를 통해 처리
    def forward(self, x):
        x = (x + 1.0) / 2.0
        x = (x - self.mean) / self.std
        return self.vgg(x)

# UpSamplingBlock: Increases spatial dimensions of feature maps
# UpSamplingBlock: 특징 맵의 공간 차원을 증가시키는 블록
class UpSamplingBlock(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        # Convolutional layer followed by pixel shuffle for upscaling
        # 업스케일링을 위한 합성곱 층과 픽셀 셔플
        self.conv = torch.nn.Conv2d(
            in_channels=config.n_filters,
            out_channels=config.n_filters * 4,
            kernel_size=3,
            padding=1,
        )
        self.phase_shift = torch.nn.PixelShuffle(upscale_factor=2)
        self.relu = torch.nn.PReLU()

    def forward(self, x):
        return self.relu(self.phase_shift(self.conv(x)))

# ResidualBlock: Basic building block with skip connection
# ResidualBlock: 스킵 연결이 있는 기본 구성 블록
class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # First convolution block
        # 첫 번째 합성곱 블록
        self.conv1 = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = torch.nn.InstanceNorm2d(out_channels)
        self.relu1 = torch.nn.PReLU()
        
        # Second convolution block
        # 두 번째 합성곱 블록
        self.conv2 = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = torch.nn.InstanceNorm2d(out_channels)

    # Forward pass with residual connection
    # 잔차 연결이 있는 순전파
    def forward(self, x):
        y = self.relu1(self.bn1(self.conv1(x)))
        return self.bn2(self.conv2(y)) + x

# Generator: Main generator network for super-resolution
# Generator: 초해상도를 위한 주요 생성자 네트워크
class Generator(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        # Initial feature extraction
        # 초기 특징 추출
        self.neck = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=config.n_filters, kernel_size=3, padding=1),
            torch.nn.PReLU(),
        )
        
        # Main residual blocks
        # 주요 잔차 블록들
        self.stem = torch.nn.Sequential(
            *[
                ResidualBlock(in_channels=config.n_filters, out_channels=config.n_filters)
                for _ in range(config.n_layers)
            ]
        )

        # Global skip connection path
        # 전역 스킵 연결 경로
        self.bottleneck = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=config.n_filters,
                out_channels=config.n_filters,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            torch.nn.InstanceNorm2d(config.n_filters),
        )

        # Upsampling blocks for increasing resolution
        # 해상도 증가를 위한 업샘플링 블록
        self.upsampling = torch.nn.Sequential(
            UpSamplingBlock(config),
            UpSamplingBlock(config),
        )

        # Final output layer
        # 최종 출력 층
        self.head = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=config.n_filters,
                out_channels=3,
                kernel_size=3,
                padding=1,
            ),
            torch.nn.Tanh(),
        )

    def forward(self, x):
        residual = self.neck(x)
        x = self.stem(residual)
        x = self.bottleneck(x) + residual
        x = self.upsampling(x)
        return self.head(x)

# SimpleBlock: Basic discriminator building block
# SimpleBlock: 판별자의 기본 구성 블록
class SimpleBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            stride=stride,
            bias=False,
        )
        self.bn = torch.nn.InstanceNorm2d(out_channels)
        self.act = torch.nn.LeakyReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

# Discriminator: Network to distinguish between real and generated images
# Discriminator: 실제 이미지와 생성된 이미지를 구분하는 네트워크
class Discriminator(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Initial feature extraction
        # 초기 특징 추출
        self.neck = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=config.n_filters, kernel_size=3, padding=1),
            torch.nn.LeakyReLU(negative_slope=0.2),
        )

        # Main discriminator architecture with increasing channels and decreasing spatial dimensions
        # 채널은 증가하고 공간 차원은 감소하는 주요 판별자 구조
        layers = [
            SimpleBlock(
                in_channels=config.n_filters,
                out_channels=config.n_filters,
                stride=2,
            ),
            SimpleBlock(
                in_channels=config.n_filters,
                out_channels=config.n_filters * 2,
                stride=1,
            ),
            SimpleBlock(
                in_channels=config.n_filters * 2,
                out_channels=config.n_filters * 2,
                stride=2,
            ),
            SimpleBlock(
                in_channels=config.n_filters * 2,
                out_channels=config.n_filters * 4,
                stride=1,
            ),
            SimpleBlock(
                in_channels=config.n_filters * 4,
                out_channels=config.n_filters * 4,
                stride=2,
            ),
            SimpleBlock(
                in_channels=config.n_filters * 4,
                out_channels=config.n_filters * 8,
                stride=1,
            ),
            SimpleBlock(
                in_channels=config.n_filters * 8,
                out_channels=config.n_filters * 8,
                stride=2,
            ),
            # Final layer for binary classification
            # 이진 분류를 위한 최종 층
            torch.nn.Conv2d(
                in_channels=config.n_filters * 8, out_channels=1, kernel_size=1, padding=0, stride=1
            ),
        ]

        self.stem = torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.neck(x)
        return self.stem(x)