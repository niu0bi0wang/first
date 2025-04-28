import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from contextlib import nullcontext
from tqdm import tqdm

from bands_selection import BandSelector as bands_selection  # 波段选择
from wtsa import swintransformer  # window topk sparse attention
from gap import GatedAttentionPool as gap  # gated attention pooling


class CustomImageDataset(Dataset):
    """高效的遥感影像数据集"""
    def __init__(self, img_dir, transform=True, cache_size=100):
        self.img_dir = img_dir
        self.bands_name = [d for d in os.listdir(img_dir) if os.path.isdir(os.path.join(img_dir, d)) 
                           and d != 'label']
        self.label_dir = os.path.join(img_dir, 'label')
        
        # 获取文件路径并预先排序以确保一致性
        self.img_path = []
        for band in self.bands_name:
            bands_dir = os.path.join(img_dir, band)
            img_files = sorted(os.listdir(bands_dir))  # 排序确保一致性
            self.img_path.append([os.path.join(bands_dir, img) for img in img_files])
        
        self.label_paths = [os.path.join(self.label_dir, img) for img in sorted(os.listdir(self.label_dir))]
        
        # LRU缓存以提高训练期间的效率，限制大小避免内存溢出
        self.cache = {}
        self.cache_size = cache_size
        self.cache_keys = []
        
        # 数据转换
        if transform:
            self.transform_img = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])  # 标准化提高训练稳定性
            ])
            self.transform_label = transforms.ToTensor()
        else:
            self.transform_img = transforms.ToTensor()
            self.transform_label = transforms.ToTensor()
       
    def __len__(self):
        return len(self.img_path[0])
    
    def _add_to_cache(self, idx, item):
        """添加到LRU缓存"""
        if len(self.cache) >= self.cache_size:
            # 移除最老的缓存项
            oldest_key = self.cache_keys.pop(0)
            del self.cache[oldest_key]
        
        self.cache[idx] = item
        self.cache_keys.append(idx)
    
    def __getitem__(self, idx):
        # 检查缓存中是否有已加载的数据
        if idx in self.cache:
            # 更新LRU顺序
            self.cache_keys.remove(idx)
            self.cache_keys.append(idx)
            return self.cache[idx]
            
        band_images = []
        for bands_path in self.img_path:
            img_path = bands_path[idx]
            image = Image.open(img_path).convert('L')
            image_array = np.array(image)
            if self.transform_img:
                image_array = self.transform_img(image_array)
            band_images.append(image_array)
        
        img = torch.stack(band_images, dim=0)  # 直接获得7通道图像 [7, H, W]

        label_path = self.label_paths[idx]
        label = Image.open(label_path).convert('L')
        label_array = np.array(label)
        if self.transform_label:
            label_array = self.transform_label(label_array).float()  # 使用float避免类型错误
        
        result = {'image': img, 'label': label_array}
        # 添加到缓存
        self._add_to_cache(idx, result)
        return result


class DoubleConv(nn.Module):
    """双卷积块，可选下采样"""
    def __init__(self, in_channels, out_channels, downsample=True):
        super().__init__()
        stride = 2 if downsample else 1
        self.down_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        return self.down_conv(x)


class UNet(nn.Module):
    """增强U-Net模型，包含带选择器、注意力门控池化和窗口稀疏注意力"""
    def __init__(self, in_channels=7, selected_bands=3, out_channels=1):
        super(UNet, self).__init__()
        self.bands_selection = bands_selection(num_bands=in_channels, selected_bands=selected_bands)
        
        # 门控注意力池化
        self.pool1 = gap(32)
        self.pool2 = gap(64)
        self.pool3 = gap(128)
        self.pool4 = gap(256)
        
        # U-Net编码器路径
        self.down1 = DoubleConv(selected_bands, 32)
        self.down2 = DoubleConv(32, 64)
        self.down3 = DoubleConv(64, 128)
        self.down4 = DoubleConv(128, 256, downsample=False)
        
        # 瓶颈层
        self.bottleneck = DoubleConv(256, 512, downsample=False)
        
        # U-Net解码器路径
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.upconv1 = DoubleConv(512, 256, downsample=False)
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upconv2 = DoubleConv(256, 128, downsample=False)
        
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.upconv3 = DoubleConv(128, 64, downsample=False)
        
        self.up4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.upconv4 = DoubleConv(64, 32, downsample=False)
        
        # 最终输出层
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=1)
        )
        
        # 初始化权重
        self._initialize_weights()
        
    def _initialize_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        # 获取选定的波段及其索引
        x, selected_indices = self.bands_selection(x)
        
        # 编码器路径
        c1 = self.down1(x)
        p1 = self.pool1(c1)
        
        c2 = self.down2(p1)
        p2 = self.pool2(c2)
        
        c3 = self.down3(p2)
        p3 = self.pool3(c3)
        
        c4 = self.down4(p3)
        p4 = self.pool4(c4)
        
        # 瓶颈
        b = self.bottleneck(p4)
        
        # 解码器路径与跳跃连接
        up1 = self.up1(b)
        # 确保特征图尺寸匹配
        if up1.size() != c4.size():
            up1 = F.interpolate(up1, size=c4.size()[2:], mode='bilinear', align_corners=False)
        
        # 应用Swin Transformer处理跳跃连接特征
        c4_att = swintransformer(c4, dim=256)
        merged1 = torch.cat([up1, c4_att], dim=1)
        x = self.upconv1(merged1)
        
        up2 = self.up2(x)
        if up2.size() != c3.size():
            up2 = F.interpolate(up2, size=c3.size()[2:], mode='bilinear', align_corners=False)
        c3_att = swintransformer(c3, dim=128)
        merged2 = torch.cat([up2, c3_att], dim=1)
        x = self.upconv2(merged2)
        
        up3 = self.up3(x)
        if up3.size() != c2.size():
            up3 = F.interpolate(up3, size=c2.size()[2:], mode='bilinear', align_corners=False)
        c2_att = swintransformer(c2, dim=64)
        merged3 = torch.cat([up3, c2_att], dim=1)
        x = self.upconv3(merged3)
        
        # 最终输出
        output = self.final_conv(x)
        
        return output


def train_model(model, train_dataloader, test_dataloader, criterion, optimizer, scheduler,
                device, num_epochs=50, grad_clip_value=1.0, mixed_precision=True):
    """
    优化的训练流程，包含混合精度训练、梯度裁剪和进度跟踪
    """
    best_iou = 0.0
    history = {'train_loss': [], 'val_loss': [], 'precision': [], 'recall': [], 'f1': [], 'iou': []}
    scaler = torch.cuda.amp.GradScaler(enabled=mixed_precision)
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        
        for batch in train_bar:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            # 使用自动混合精度
            with torch.cuda.amp.autocast(enabled=mixed_precision):
                outputs = model(images)
                
                # 确保标签和输出的形状相匹配
                if outputs.shape != labels.shape:
                    labels = F.interpolate(labels, size=outputs.shape[2:], mode='nearest')
                    
                loss = criterion(outputs, labels)
            
            # 梯度更新
            optimizer.zero_grad()
            
            if mixed_precision:
                scaler.scale(loss).backward()
                # 梯度裁剪防止梯度爆炸
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)
                optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            train_bar.set_postfix(loss=loss.item())
        
        avg_train_loss = train_loss / len(train_dataloader.dataset)
        history['train_loss'].append(avg_train_loss)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        val_bar = tqdm(test_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Valid]")
        
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=mixed_precision):
            for batch in val_bar:
                images = batch['image'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(images)
                
                # 确保标签和输出的形状相匹配
                if outputs.shape != labels.shape:
                    labels = F.interpolate(labels, size=outputs.shape[2:], mode='nearest')
                
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                val_bar.set_postfix(loss=loss.item())
                
                # 预测转为二值
                preds = (torch.sigmoid(outputs) > 0.5).float()
                all_preds.append(preds.cpu())
                all_targets.append(labels.cpu())
        
        # 合并所有验证数据
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        
        # 计算指标
        precision, recall, f1, iou = calculate_metrics(all_preds, all_targets)
        avg_val_loss = val_loss / len(test_dataloader.dataset)
        
        # 学习率调整
        scheduler.step(avg_val_loss)
        
        # 记录历史
        history['val_loss'].append(avg_val_loss)
        history['precision'].append(precision)
        history['recall'].append(recall)
        history['f1'].append(f1)
        history['iou'].append(iou)
        
        # 输出结果
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"Precision: {precision:.4f} | Recall: {recall:.4f}")
        print(f"F1-score: {f1:.4f} | mIoU: {iou:.4f}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 保存最佳模型
        if iou > best_iou:
            best_iou = iou
            print(f"新的最佳mIoU: {best_iou:.4f}，保存模型...")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'iou': best_iou,
                'history': history
            }, "best_model.pth")
    
    return model, history


def calculate_metrics(preds, targets):
    """计算分割评估指标"""
    preds = preds.view(-1).cpu().numpy()
    targets = targets.view(-1).cpu().numpy()
    
    # 计算真正例、假正例、假负例和真负例
    TP = ((preds == 1) & (targets == 1)).sum()
    FP = ((preds == 1) & (targets == 0)).sum()
    FN = ((preds == 0) & (targets == 1)).sum()
    TN = ((preds == 0) & (targets == 0)).sum()
    
    # 计算评估指标，增加稳定性
    epsilon = 1e-7
    precision = TP / (TP + FP + epsilon)
    recall = TP / (TP + FN + epsilon)
    f1 = 2 * precision * recall / (precision + recall + epsilon)
    iou = TP / (TP + FP + FN + epsilon)
    
    return precision, recall, f1, iou


if __name__ == "__main__":
    # 设置随机种子以提高可重复性
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # 设置路径和数据加载
    train_path = r"C:\Users\27268\Desktop\item\optical\train"
    test_path = r"C:\Users\27268\Desktop\item\optical\test"

    # 优化数据加载，使用多工作线程加快数据加载
    num_workers = min(8, os.cpu_count() or 1) if torch.cuda.is_available() else 0
    
    # 创建数据集，对训练和测试使用相同的转换以提高一致性
    train_dataset = CustomImageDataset(train_path, transform=True, cache_size=100)
    test_dataset = CustomImageDataset(test_path, transform=True, cache_size=100)
    
    # 批次大小设置，根据可用GPU内存调整
    batch_size = 16
    
    # 创建DataLoader
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),  # 数据直接加载到GPU内存
        drop_last=True  # 丢弃不完整的最后一批，避免批归一化问题
    )
    
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,  # 不打乱测试数据顺序
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建模型
    model = UNet(in_channels=7, selected_bands=3, out_channels=1).to(device)
    
    # 打印模型结构
    print(f"模型架构: {model.__class__.__name__}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")
    
    # 损失函数和优化器
    # 使用组合损失函数以提高性能
    class CombinedLoss(nn.Module):
        def __init__(self, bce_weight=0.8, dice_weight=0.2):
            super(CombinedLoss, self).__init__()
            self.bce_weight = bce_weight
            self.dice_weight = dice_weight
            self.bce = nn.BCEWithLogitsLoss()
            
        def forward(self, pred, target):
            bce_loss = self.bce(pred, target)
            
            # Dice Loss
            pred_sigmoid = torch.sigmoid(pred)
            smooth = 1.0
            intersection = (pred_sigmoid * target).sum()
            dice_loss = 1 - (2. * intersection + smooth) / (pred_sigmoid.sum() + target.sum() + smooth)
            
            # 组合损失
            return self.bce_weight * bce_loss + self.dice_weight * dice_loss

    # 使用更强大的损失函数
    criterion = CombinedLoss()
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True, min_lr=1e-7
    )
    
    # 训练模型
    print("开始训练...")
    model, history = train_model(
        model, train_dataloader, test_dataloader, 
        criterion, optimizer, scheduler, device, 
        num_epochs=50, grad_clip_value=1.0, mixed_precision=True
    )
    
    # 保存最终模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'history': history
    }, "final_model.pth")
    
    print("训练完成!")
    
    # 测试模型输出形状
    test_input = torch.randn(1, 7, 128, 128).to(device)
    with torch.no_grad(), torch.cuda.amp.autocast():
        output = model(test_input)
    print(f"输出形状: {output.shape}")