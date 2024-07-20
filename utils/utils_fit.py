import os

import torch
from nets.unet_training import CE_Loss, Dice_loss, Focal_Loss
from tqdm import tqdm

from utils.utils import get_lr
from utils.utils_metrics import f_score


def fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, 
                  epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, 
                  dice_loss, focal_loss, cls_weights, num_classes, fp16, scaler, 
                  save_period, save_dir, model_dir, model_best_dir, local_rank=0):
    total_loss      = 0
    total_f_score   = 0

    val_loss        = 0
    val_f_score     = 0

    # 当前epoch的训练过程
    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
        # Sets the module in training mode
        model_train.train()
    # gen:训练集DataLoader，epch_step:一个epoch需要进行的步数(num_train//batch_size)
    for iteration, batch in enumerate(gen):
        # iteration为迭代轮次0,1,2,3,4……
        # batch是list，len由dataloader.py中的UnetDataset类的__getitem__返回值个数决定
        # 在此batch的len为3，batch[0]是batch_size个图像，batch[1]是对应的batch_size个png分割label
        # batch[2]是png的one-hot形式，具体作用待解析？
        
        if iteration >= epoch_step: 
            break
        # 从gen获取的一个batch
        imgs, pngs, labels = batch
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs    = imgs.cuda(local_rank)
                pngs    = pngs.cuda(local_rank)
                labels  = labels.cuda(local_rank)
                weights = weights.cuda(local_rank)

        optimizer.zero_grad()
        # 混合精度训练，可节约缓存，默认设置False
        if not fp16:
            #----------------------#
            #   前向传播
            #----------------------#
            outputs = model_train(imgs)
            #----------------------#
            #   损失计算
            #----------------------#
            ce_loss = CE_Loss(outputs, pngs, weights, num_classes = num_classes)

            if focal_loss:
                f_loss = Focal_Loss(outputs, pngs, weights, num_classes = num_classes)
            else:
                f_loss = 0

            if dice_loss:
                d_loss = Dice_loss(outputs, labels)
            else:
                d_loss = 0

            loss = ce_loss + f_loss + d_loss
            
            with torch.no_grad():
                #-------------------------------#
                #   计算f_score
                #-------------------------------#
                _f_score = f_score(outputs, labels)

            loss.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                #----------------------#
                #   前向传播
                #----------------------#
                outputs = model_train(imgs)
                #----------------------#
                #   损失计算
                #----------------------#
                ce_loss = CE_Loss(outputs, pngs, weights, num_classes = num_classes)

                if focal_loss:
                    f_loss = Focal_Loss(outputs, pngs, weights, num_classes = num_classes)
                else:
                    f_loss = 0

                if dice_loss:
                    d_loss = Dice_loss(outputs, labels)
                else:
                    d_loss = 0

                loss = ce_loss + f_loss + d_loss

                with torch.no_grad():
                    #-------------------------------#
                    #   计算f_score
                    #-------------------------------#
                    _f_score = f_score(outputs, labels)

            #----------------------#
            #   反向传播
            #----------------------#
            # 用于混合精度训练的方法，该方法在训练网络时将单精度（FP32）与半精度(FP16)结合在一起，
            # 并使用相同的超参数实现了与FP32几乎相同的精度，加快训练和推断的速度
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
        # 从包含单个元素的张量中取出该元素值，并保持该元素的类型不变
        # loss类型为torch.Tensor,total_loss类型为float
        # total_loss
        total_loss      += loss.item()
        # print(type(loss))
        # print(type(total_loss))
        total_f_score   += _f_score.item()
        
        if local_rank == 0:
            # total_loss取的是均值，与batch_size大小无关
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1), 'f_score': total_f_score / (iteration + 1), 'lr': get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)

    # 当前epoch的验证过程
    # eval():Sets the module in evaluation mode.
    model_train.eval()
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        # 从gen_val获取的一个batch
        imgs, pngs, labels = batch
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs    = imgs.cuda(local_rank)
                pngs    = pngs.cuda(local_rank)
                labels  = labels.cuda(local_rank)
                weights = weights.cuda(local_rank)

            #----------------------#
            #   前向传播
            #----------------------#
            outputs = model_train(imgs)
            #----------------------#
            #   损失计算
            #----------------------#
            ce_loss = CE_Loss(outputs, pngs, weights, num_classes = num_classes)

            if focal_loss:
                f_loss = Focal_Loss(outputs, pngs, weights, num_classes = num_classes)
            else:
                f_loss = 0

            if dice_loss:
                d_loss = Dice_loss(outputs, labels)
            else:
                d_loss = 0

            loss = ce_loss + f_loss + d_loss
            #-------------------------------#
            #   计算f_score
            #-------------------------------#
            _f_score    = f_score(outputs, labels)

            val_loss    += loss.item()
            val_f_score += _f_score.item()
            
        if local_rank == 0:
            pbar.set_postfix(**{'val_loss'  : val_loss / (iteration + 1),
                                'f_score'   : val_f_score / (iteration + 1),
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)
            
    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        loss_history.append_loss(epoch + 1, total_loss/ epoch_step, val_loss/ epoch_step_val)
        # eval_callback是EvalCallback类(在callbacks.py中定义)的一个实例，
        # on_epoch_end是该类的一个方法
        eval_callback.on_epoch_end(epoch + 1, model_train)
        print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))
        
        #-----------------------------------------------#
        #   保存权值
        #-----------------------------------------------#
        # epoch为当前epoch数（从0开始计数），Epoch是总epoch数，即train.py中的UnFreeze_Epoch，一般设置成100
        if ((epoch + 1) % save_period == 0 or epoch + 1 == Epoch) and Epoch-epoch <= 31 :
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            torch.save(model.state_dict(), os.path.join(model_dir, 'ep%03d-loss%.3f-val_loss%.3f.pth'%((epoch + 1), total_loss / epoch_step, val_loss / epoch_step_val)))
        
        # 有些模型训练完成后并未保存最优参数，将info纪录在txt内，查看情况
        if Epoch-epoch <=500:
            if not os.path.exists(model_best_dir):
                os.makedirs(model_best_dir)
            bestepoch_finfo = os.path.join(model_best_dir,"last50_epochinfo .txt")
            # 将当前最佳轮次信息纪录下来
            with open (bestepoch_finfo,'a') as f:
                f.write("------------------------------------------------------------------\n")
                f.write("Epoch: " + str(Epoch) + "\n")
                f.write("epoch: " + str(epoch) + "\n")
                f.write("len(loss_history.val_loss): " + str(len(loss_history.val_loss)) + "\n")
                f.write("val_loss/epoch_step_val" + str(val_loss/epoch_step_val) + "\n")
                f.write("min(loss_history.val_loss): " + str(min(loss_history.val_loss)) + "\n")
                f.write("\n")

        if Epoch-epoch<=50 and (len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss)) :            
            print('Save best model to %s\\best_epoch_weights.pth' % (model_best_dir))
            if not os.path.exists(model_best_dir):
                os.makedirs(model_best_dir)
            torch.save(model.state_dict(), os.path.join(model_best_dir, "best_epoch_weights.pth"))
            fname = os.path.join(model_best_dir,"best_epoch_info.txt")
            # 将当前最佳轮次信息纪录下来
            with open (fname,'w') as f:
                f.write("epoch: " + str(epoch+1) + "\n")
                f.write('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))
                f.write("\n")
                f.write("\n")
            
        # torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))

def fit_one_epoch_no_val(model_train, model, loss_history, optimizer, epoch, epoch_step, gen, Epoch, cuda, dice_loss, focal_loss, cls_weights, num_classes, fp16, scaler, save_period, save_dir, local_rank=0):
    total_loss      = 0
    total_f_score   = 0
    
    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step: 
            break
        imgs, pngs, labels = batch
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs    = imgs.cuda(local_rank)
                pngs    = pngs.cuda(local_rank)
                labels  = labels.cuda(local_rank)
                weights = weights.cuda(local_rank)

        optimizer.zero_grad()
        if not fp16:
            #----------------------#
            #   前向传播
            #----------------------#
            outputs = model_train(imgs)
            #----------------------#
            #   损失计算
            #----------------------#
            ce_loss = CE_Loss(outputs, pngs, weights, num_classes = num_classes)

            if focal_loss:
                f_loss = Focal_Loss(outputs, pngs, weights, num_classes = num_classes)
            else:
                f_loss = 0

            if dice_loss:
                d_loss = Dice_loss(outputs, labels)
            else:
                d_loss = 0

            loss = ce_loss + f_loss + d_loss

            with torch.no_grad():
                #-------------------------------#
                #   计算f_score
                #-------------------------------#
                _f_score = f_score(outputs, labels)

            loss.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                #----------------------#
                #   前向传播
                #----------------------#
                outputs = model_train(imgs)
                #----------------------#
                #   损失计算
                #----------------------#
                ce_loss = CE_Loss(outputs, pngs, weights, num_classes = num_classes)

                if focal_loss:
                    f_loss = Focal_Loss(outputs, pngs, weights, num_classes = num_classes)
                else:
                    f_loss = 0

                if dice_loss:
                    d_loss = Dice_loss(outputs, labels)
                else:
                    d_loss = 0

                loss = ce_loss + f_loss + d_loss

                with torch.no_grad():
                    #-------------------------------#
                    #   计算f_score
                    #-------------------------------#
                    _f_score = f_score(outputs, labels)

            #----------------------#
            #   反向传播
            #----------------------#
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss      += loss.item()
        total_f_score   += _f_score.item()
        
        if local_rank == 0:
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1), 
                                'f_score'   : total_f_score / (iteration + 1),
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        loss_history.append_loss(epoch + 1, total_loss/ epoch_step)
        print('Epoch:'+ str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f' % (total_loss / epoch_step))
        
        #-----------------------------------------------#
        #   保存权值
        #-----------------------------------------------#
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(model.state_dict(), os.path.join(save_dir, 'ep%03d-loss%.3f.pth'%((epoch + 1), total_loss / epoch_step)))

        if len(loss_history.losses) <= 1 or (total_loss / epoch_step) <= min(loss_history.losses):
            print('Save best model to best_epoch_weights.pth')
            torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))
            
        torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))