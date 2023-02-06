import torch

from mico_competition import load_cifar10, load_purchase100, load_sst2


def load_data(dataset_name):
    if dataset_name == 'cifar10':
        return load_cifar10(dataset_dir='./data/data')
    elif dataset_name == 'purchase100':
        return load_purchase100(dataset_dir='./data/data')
    elif dataset_name == 'sst2':
        return load_sst2()
    else:
        raise ValueError("dataset_name must be one of {'cifar10', 'purchase100', 'sst2'}")


class Parallel(object):

    def __init__(self, num_gpus, jobs_per_gpu):
        if torch.cuda.device_count() < num_gpus:
            raise RuntimeError('No enough gpus are available')
        torch.multiprocessing.set_sharing_strategy('file_system')
        self.context = torch.multiprocessing.get_context('spawn')
        self.devices = [torch.device(f'cuda:{i}') for i in range(num_gpus)] * jobs_per_gpu

    def __call__(self, func, tasks, args, collect_func=None):
        if len(self.devices) == 1:
            print(f'[{self.devices[0]}] num tasks allocated: {len(tasks)}')
            return func(tasks, *args, self.devices[0])
        # parallel tasks
        if len(tasks) == 0:
            raise RuntimeError('Task list is empty')
        # load balance
        nums = [len(tasks) // len(self.devices)] * len(self.devices)
        for i in range(len(tasks) % len(self.devices)):
            nums[i] += 1
        cumsums = torch.cumsum(torch.tensor(nums), dim=0).tolist()
        cumsums = [0] + cumsums
        # parallel
        pool = self.context.Pool(len(self.devices))
        result_list = []
        for i, device in enumerate(self.devices):
            sub_tasks = tasks[cumsums[i]:cumsums[i + 1]]
            if len(sub_tasks) > 0:
                result = pool.apply_async(func, args=(sub_tasks, *args, device))
                result_list.append(result)
            print(f'[{device}] num tasks allocated: {len(sub_tasks)}')
        pool.close()
        pool.join()

        # collect result
        if collect_func is None:
            out = []
            for result in result_list:
                result = result.get()
                if isinstance(result, list):
                    out.extend(result)
                else:
                    out.append(result)
            return out
        else:
            return collect_func(result_list)
