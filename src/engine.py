import torch
import torch.nn.functional as F

def topk_acc(logits, y, k):
    return (logits.topk(k, dim=1).indices.eq(y.view(-1,1))).any(dim=1).float().mean().item()

def train_one_epoch(model, loader, optimizer, scaler, device):
    model.train()
    total_loss = 0.0; total_correct = 0; total = 0
    amp = torch.cuda.is_available()
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp):
            logits = model(x)
            loss = F.cross_entropy(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        bs = x.size(0)
        total_loss += loss.item() * bs
        total_correct += (logits.argmax(1) == y).sum().item()
        total += bs
    return {"loss": total_loss/total, "acc": total_correct/total}

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss_sum=0.0; n=0; t1=t5=0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
        bs = x.size(0); n += bs
        t1 += topk_acc(logits, y, 1) * bs
        t5 += topk_acc(logits, y, 5) * bs
    return {"loss": loss_sum/n, "top1": t1/n, "top5": t5/n}
