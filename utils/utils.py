from __future__ import print_function

from collections import defaultdict, deque
import datetime
import pickle
import time

import torch
import torch.distributed as dist

import errno
import os


def calculate_metrics(predictions, ground_truths, iou_threshold=0.2):
    """
    Calculate precision, recall, F1-score, and IoU score for overall and each class in a set of predictions and ground truth boxes.
    """
    class_stats = {}
    
    for pred_boxes, gt_boxes in zip(predictions, ground_truths):
        matched_gt = {i: False for i in range(len(gt_boxes))}
        #print(f'pred_boxes:{pred_boxes}, gt_boxes : {gt_boxes}')
        if len(pred_boxes)==0 and len(gt_boxes)==0:
          update_class_stats(class_stats,0,'TP')
          continue
        if not gt_boxes and pred_boxes:
          
          update_class_stats(class_stats,0,'FN')
          continue

        if not pred_boxes and gt_boxes:
          
          update_class_stats(class_stats,0,'FP')
          continue

        for pred_class, pred_box in pred_boxes:
            best_iou = 0
            best_match = None
            
            for i, (gt_class, gt_box) in enumerate(gt_boxes):
               # pred_class=pred_class.item()
               # gt_class=gt_class.item()
                if isinstance(pred_class,torch.Tensor):
                  pred_class=pred_class.item()
                if isinstance(gt_class,torch.Tensor):
                  gt_class=gt_class.item()


                if pred_class != gt_class: # class 불일치할경우 넘기기
                    #print(f'틀렸음 pred_class : {pred_class}, gt_class : {gt_class}')
                    continue
                #print(f'라벨은 맞음 pred_class, gt_class :{pred_class},{gt_class}')
                iou = calculate_iou(pred_box, gt_box) # 클래스 일치할 경우 box iou계산
                #print(f'{pred_class}에 대한 iou :{iou}')
                if iou > best_iou:
                    best_iou = iou
                    best_match = i
            
            # 모든 gt를 돌고 난 후 best_iou > iou_threshold이고 best_match일 경우
            # 1) 해당 gt가 best_match가 없을 경우 
            if best_iou > iou_threshold and best_match is not None: 
                if not matched_gt[best_match]:
                    update_class_stats(class_stats, pred_class, 'TP', best_iou)
                    matched_gt[best_match] = True
                else:
                    update_class_stats(class_stats, pred_class, 'FP') # 이미 임자가 있는데 잘못고름
            else:
                update_class_stats(class_stats, pred_class, 'FP') 

        for i, (gt_class, _) in enumerate(gt_boxes):
            if not matched_gt[i]:
                update_class_stats(class_stats, gt_class, 'FN') 

    print(f'class_stats : {class_stats}')
    return calculate_classwise_metrics(class_stats)

def update_class_stats(stats, cls, update_type, iou_score=0):
    if cls not in stats:
        stats[cls] = {'TP': 0, 'FP': 0, 'FN': 0, 'total_iou': 0}
    
    if update_type == 'TP':
        stats[cls]['TP'] += 1
        stats[cls]['total_iou'] += iou_score
    elif update_type == 'FP':
        stats[cls]['FP'] += 1
    elif update_type == 'FN':
        stats[cls]['FN'] += 1

def calculate_classwise_metrics(stats):
    class_metrics = {}
    total_TP, total_FP, total_FN, total_iou = 0, 0, 0, 0
    for cls, counts in stats.items():
        precision = counts['TP'] / (counts['TP'] + counts['FP']) if (counts['TP'] + counts['FP']) > 0 else 0
        recall = counts['TP'] / (counts['TP'] + counts['FN']) if (counts['TP'] + counts['FN']) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        average_iou = counts['total_iou'] / counts['TP'] if counts['TP'] > 0 else 0

        class_metrics[cls] = {'precision': precision, 'recall': recall, 'f1_score': f1_score, 'average_iou': average_iou}

        total_TP += counts['TP']
        total_FP += counts['FP']
        total_FN += counts['FN']
        total_iou += counts['total_iou']

    # 전체 성능 계산
    total_precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0
    total_recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0
    if total_precision+total_recall==0:
      total_f1_score=0
    else:
      total_f1_score = 2 * total_precision * total_recall / (total_precision + total_recall) if (total_precision + total_recall) > 0 else 0
    total_average_iou = total_iou / total_TP if total_TP > 0 else 0

    return {'total': {'precision': total_precision, 'recall': total_recall, 'f1_score': total_f1_score, 'average_iou': total_average_iou}, 'per_class': class_metrics}



# IoU 계산 함수
def calculate_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    """
    
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2
    
    # Calculate area of intersection
    xi1 = max(x1, x1g)
    yi1 = max(y1, y1g)
    xi2 = min(x2, x2g)
    yi2 = min(y2, y2g)
    intersection_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)
    
    # Calculate area of union
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2g - x1g) * (y2g - y1g)
    union_area = box1_area + box2_area - intersection_area
    
    # Calculate IoU
    iou = intersection_area / union_area if union_area != 0 else 0
    
    return iou
    

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device="cuda")
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device="cuda"))
    if local_size != max_size:
        padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device="cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = self.delimiter.join([
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}',
            'max mem: {memory:.0f}'
        ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                print(log_msg.format(
                    i, len(iterable), eta=eta_string,
                    meters=str(self),
                    time=str(iter_time), data=str(data_time),
                    memory=torch.cuda.max_memory_allocated() / MB))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def collate_fn(batch):
    return tuple(zip(*batch))


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):

    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)
