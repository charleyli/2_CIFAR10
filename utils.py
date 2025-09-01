import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
import pandas as pd
def train_model(model, device, train_loader, optimizer,epoch):
    # 模型的训练模式
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
    # for batch_index, (data, target) in enumerate(train_loader):
    for data, target in pbar:
        data = data.to(device)
        target = target.to(device)
        # print("utils.py 16 data.shape in enumerate(train_loader)",data.shape,batch_index) 
        # print("utils.py 17 target.shape in enumerate(train_loader)", target.shape,batch_index)
        output = model(data)
        optimizer.zero_grad()
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        
        # loss.item() 将这个张量转换成 Python 的浮点数 
        # data.size(0)是batch_size
        running_loss = running_loss + loss.item() * data.size(0) 
        # output 的形状一般是 [batch_size, num_classes]，表示每个样本每个类别的预测得分（logits）
        # _ 是最大值本身（我们不需要）
        # pred 是最大值所在的索引，即模型预测的类别。
        _, pred = output.max(1)
        # pred.eq(target)返回一个布尔张量，表示每个预测是否正确。
        correct = correct + pred.eq(target).sum().item()
        total = total + target.size(0) # 当前批次的样本数

        pbar.set_postfix(loss=loss.item(), acc=100.*correct/total)

    avg_loss = running_loss / len(train_loader.dataset) # len(train_loader.dataset) 是训练集总样本数
    accuracy = 100.* correct / total
    print(f"epoch: {epoch}")
    print(f"Train Loss: {avg_loss:.4f}, Train Accuracy: {accuracy:.2f}%")
    return avg_loss, accuracy



def test_model(model, device, test_loader):
    # 模型验证
    model.eval()
    # 正确率
    correct = 0.0
    # 损失
    test_loss = 0.0
    with torch.no_grad(): # 不计算梯度，也不进行反向传播
        for data, target in  test_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data) # output 形状一般为 [batch_size, num_classes]，表示每个样本各类别的预测分数（logits）

            # reduction='sum'：对批次内所有样本的损失求和，而不是平均。
            # .item()：将张量转为 Python 浮点数。
            test_loss = test_loss + F.cross_entropy(output, target, reduction='sum').item()


            # 第二种方式：pred = output.argmax(dim=1)
            # _ 是最大值本身（我们不需要）
            # pred 是最大值所在的索引，即模型预测的类别。
            _,pred = output.max(1)
            pred.eq(target)



            # pred.eq(target)：返回布尔张量，每个元素表示预测是否正确。
            # .sum()：对布尔张量求和，得到本批次预测正确的样本数。
            # .item()：转换为 Python 整数。
            correct = correct + pred.eq(target).sum().item()
    test_loss = test_loss / len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset) * 100
    print("Test Loss : {:.6f} Test Accuracy : {:.2f}%\n".format(test_loss,accuracy))
    return test_loss, accuracy


def write2csv(epoch,train_loss, train_acc,test_loss, test_acc):
    log_file = "log.csv"
    columns = ["epoch","Train Loss", "Train Accuracy", "Test Loss", "Test Accuracy"]

    # 保留两位小数
    train_loss = round(train_loss, 2)
    train_acc = round(train_acc, 2)
    test_loss = round(test_loss, 2)
    test_acc = round(test_acc, 2)

    # 生成 DataFrame
    df = pd.DataFrame([[epoch, train_loss, train_acc, test_loss, test_acc]], columns=columns)

    if not os.path.exists(log_file):  
        # 文件不存在 -> 新建文件并写入表头
        df.to_csv(log_file, mode='w', index=False, header=True)
    else:
        # 文件存在 -> 追加数据，不写表头
        df.to_csv(log_file, mode='a', index=False, header=False)