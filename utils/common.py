import os, torch, random, time, wandb, yaml
import numpy as np


project_name = os.path.basename(os.getcwd())


def save_val_log(name, iou_list, train_dataset, tf_writer, logger, cur_iter):
        tb_name = 'valie_' + name + "/"
        s = name + ' IoU: \n'
        for ci, iou_tmp in enumerate(iou_list):
            s += '{:5.2f} '.format(100 * iou_tmp)
            class_name = train_dataset.label_name[ci]
            s += ' ' + class_name + ' '
            tf_writer.add_scalar(tb_name + class_name, 100 * iou_tmp, cur_iter)
        logger.info(s)


def save_best_check(net_G, net_D,
                    G_optim, D_optim, src_centers,
                    cur_iter, logger, log_dir, name, iou):
    logger.info('**** Best mean {} val iou:{:.1f} ****'.format(name, iou * 100))
    filename = 'checkpoint_val_' + name + '.tar'
    fname = os.path.join(log_dir, filename)
    save_checkpoint(fname, net_G, net_D, 
                    G_optim, D_optim, src_centers, cur_iter)


def save_checkpoint(fname, net_G, net_D, 
                    G_optim, D_optim,
                    src_centers, cur_iter):
    save_dict = {
        'cur_iter': cur_iter + 1,  # after training one epoch, the start_epoch should be epoch+1
        'G_optim_state_dict': G_optim.state_dict(),
    }
    if D_optim is not None:
        save_dict['D_optim_state_dict'] =  D_optim.state_dict()

    if src_centers is not None:
        save_dict['src_centers_Proto'] = src_centers.Proto
        save_dict['src_centers_Amount'] = src_centers.Amount
    # with nn.DataParallel() the net is added as a submodule of DataParallel
    try:
        save_dict['model_state_dict'] = net_G.module.state_dict()
        if net_D is not None:
            save_dict['D_out_model_state_dict'] = net_D.module.state_dict()
    except AttributeError:
        save_dict['model_state_dict'] = net_G.state_dict()
        if net_D is not None:
            save_dict['D_out_model_state_dict'] = net_D.state_dict()

    torch.save(save_dict, fname)



def loadCheckPoint(CHECKPOINT_PATH, net, classifier, D_out,
                    G_optim, head_optim, D_out_optim):

    checkpoint = torch.load(CHECKPOINT_PATH)

    c_iter = checkpoint['epoch']

    net.load_state_dict(checkpoint['model_state_dict'])
    classifier.load_state_dict(checkpoint['classifier_state_dict'])
    D_out.load_state_dict(checkpoint['D_out_model_state_dict'])

    G_optim.load_state_dict(checkpoint['G_optim_state_dict'])
    head_optim.load_state_dict(checkpoint['head_optim_state_dict'])
    D_out_optim.load_state_dict(checkpoint['D_out_optim_state_dict'])

    return net, classifier, D_out, G_optim, head_optim, D_out_optim, c_iter

def clean_summary(filesuammry):
    """
    remove keys from wandb.log()
    Args:
        filesuammry:

    Returns:

    """
    keys = [k for k in filesuammry.keys() if not k.startswith('_')]
    for k in keys:
        filesuammry.__delitem__(k)
    return filesuammry

def classProperty2dict(obj):
    pr = {}
    for name in dir(obj):
        value = getattr(obj, name)
        if not name.startswith('__') and not callable(value):
            pr[name] = value
    return pr


def make_reproducible(iscuda=True, seed=999):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if iscuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # set True will make data load faster
        #   but, it will influence reproducible
        # torch.backends.cudnn.benchmark = True
        # torch.backends.cudnn.deterministic = True

def mkdir(dir):
    if not os.path.isdir(dir):
        os.makedirs(dir, exist_ok=False)

def torch_set_gpu(gpus):
    if type(gpus) is int:
        gpus = [gpus]

    cuda = all(gpu >= 0 for gpu in gpus)

    if cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(gpu) for gpu in gpus])
        assert cuda and torch.cuda.is_available(), "%s has GPUs %s unavailable" % (
            os.environ['HOSTNAME'], os.environ['CUDA_VISIBLE_DEVICES'])
        # torch.backends.cudnn.benchmark = True # speed-up cudnn
        # torch.backends.cudnn.fastest = True # even more speed-up?
        hint('Launching on GPUs ' + os.environ['CUDA_VISIBLE_DEVICES'])

    else:
        hint('Launching on CPU')

    return cuda


def hint(msg):
    timestamp = f'{time.strftime("%m/%d %H:%M:%S", time.localtime(time.time()))}'
    print('\033[1m' + project_name + ' >> ' + timestamp + ' >> ' + '\033[0m' + msg)

