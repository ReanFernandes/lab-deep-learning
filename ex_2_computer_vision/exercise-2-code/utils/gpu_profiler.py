import os
from typing import Optional, List

import nvidia_smi
import psutil


class GPUProfiler:
    """
    Get information about GPU and RAM usage as string.

    Setup:
        pip install nvidia-ml-py3 pynvml

    Notes:
        gpu_load:
            Percent of time over the past second in which any work has been executing on the GPU.
        gpu_memory_load:
            Percent of time over the past second in which any framebuffer memory has been read or stored.

    Usage:
        >>> gpu_profiler = GPUProfiler()
        >>> print(gpu_profiler.profile_to_str())

    """

    def __init__(self):
        self.gpu_count = -1
        self.gpu_handles = []
        self.init_done = False

    def start(self):
        if self.init_done:
            return
        self._start()

    def _start(self):
        nvidia_smi.nvmlInit()
        self.gpu_count = nvidia_smi.nvmlDeviceGetCount()
        if self.gpu_count == 0:
            self.gpu_count = 1
        self.gpu_handles = []
        for i in range(self.gpu_count):
            try:
                handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
            except Exception as e:
                print(f"WARNING: Could not get handle for GPU {i}: "
                      f"{e} ({e.__class__.__name__})")
                handle = None
            self.gpu_handles.append(handle)
        self.init_done = True

    def get_gpu_numbers(self):
        """
        Try to determine gpu numbers from env variables, fallback is to profile all gpus

        Returns:
            List with GPU numbers
        """
        self.start()
        gpu_numbers = list(range(self.gpu_count))
        device_env = os.getenv("CUDA_VISIBLE_DEVICES")
        if device_env is not None:
            try:
                gpu_numbers = [int(x.strip()) for x in device_env.split(",") if x.strip() != ""]
            except Exception:
                pass
            if len(gpu_numbers) == 0:
                gpu_numbers = list(range(self.gpu_count))
        return gpu_numbers

    def profile_gpu(self, gpu_numbers: Optional[List[int]] = None):
        """
        Profile GPU utilization

        Args:
            gpu_numbers: optional numbers of GPUs to profile

        Returns:
            Lists with entries for each GPU with content:
                name, total memory (GB), used memory (GB), gpu load (0-1), memory load (0-1), temperature (°C)

        """
        self.start()
        gpu_numbers = self.get_gpu_numbers() if gpu_numbers is None else gpu_numbers
        gpu_numbers = [i for i in gpu_numbers if self.gpu_handles[i] is not None]

        mem_objs = [nvidia_smi.nvmlDeviceGetMemoryInfo(self.gpu_handles[i]) for i in gpu_numbers]
        mem_total = [mem_obj.total / 1024 ** 3 for mem_obj in mem_objs]
        mem_used = [mem_obj.used / 1024 ** 3 for mem_obj in mem_objs]
        # names = [nvidia_smi.nvmlDeviceGetName(self.gpu_handles[i]).decode("utf8") for i in gpu_numbers]
        names = [nvidia_smi.nvmlDeviceGetName(self.gpu_handles[i]) for i in gpu_numbers]
        load_objs = [nvidia_smi.nvmlDeviceGetUtilizationRates(self.gpu_handles[i]) for i in
                     gpu_numbers]
        load_gpu = [load_obj.gpu / 100 for load_obj in load_objs]
        load_gpu_mem = [load_obj.memory / 100 for load_obj in load_objs]
        temp = [nvidia_smi.nvmlDeviceGetTemperature(self.gpu_handles[i],
                                                    nvidia_smi.NVML_TEMPERATURE_GPU)
                for i in gpu_numbers]
        return names, mem_total, mem_used, load_gpu, load_gpu_mem, temp

    def profile_single_gpu(self, gpu_number: int = 0):
        outputs = self.profile_gpu([gpu_number])
        return [x[0] for x in outputs]

    def profile_to_str(self, gpu_numbers: Optional[List[int]] = None):
        """
        Use profile output to create a string

        Args:
            gpu_numbers: optional numbers of GPUs to profile

        Returns:
            profile string for output
        """
        names, mem_total, mem_used, load_gpu, load_gpu_mem, temp = self.profile_gpu(gpu_numbers)
        ram_total, ram_used = profile_ram()
        # average / sum over all GPUs
        sum_mem_total: float = sum(mem_total)
        sum_mem_used: float = sum(mem_used)
        # gpu_mem_percent: float = gpu_mem_used / gpu_mem_total
        avg_load_gpu: float = sum(load_gpu) / max(1, len(load_gpu))
        avg_load_gpu_mem: float = sum(load_gpu_mem) / max(1, len(load_gpu_mem))
        avg_temp: float = sum(temp) / max(1, len(temp))

        # log the values
        # gpu_names_str = " ".join(set(names)) # throwing error saying that names is in byte format
        gpu_names_str = " ".join(set(name.decode() for name in names))
        multi_load, multi_load_mem, multi_temp, multi_mem = "", "", "", ""
        if len(load_gpu) > 1:
            multi_load = " " + ", ".join([f"{ld:.0%}" for ld in load_gpu])
            multi_load_mem = " " + ", ".join([f"{ld:.0%}" for ld in load_gpu_mem])
            multi_temp = " " + ", ".join([f"{ld:d}" for ld in temp])
            multi_mem = " " + ", ".join([f"{mem:.1f}GB" for mem in mem_used])

        out_str = (f"RAM {ram_used:.1f}/{ram_total:.1f} "
                   f"GPU {gpu_names_str} Util {avg_load_gpu:.0%}{multi_load} "
                   f"UMem {avg_load_gpu_mem:.0%}{multi_load_mem} "
                   f"Mem {sum_mem_used:.1f}/{sum_mem_total:.1f}{multi_mem} "
                   f"Temp {avg_temp:.0f}°C{multi_temp}")

        return out_str

    def check_gpus_have_errors(self, gpu_numbers: Optional[List[int]] = None):
        """
        Some GPUs can silently not work, and then jax will silently switch to CPU.
        This function checks for these errors.

        Args:
            gpu_numbers: optional numbers of GPUs to profile

        Returns:
            profile string for output
        """
        self.start()
        gpu_numbers = self.get_gpu_numbers() if gpu_numbers is None else gpu_numbers

        has_errors = False
        for i in gpu_numbers:
            if self.gpu_handles[i] is None:
                has_errors = True
                print(f"ERROR: GPU {i} handle could not be created at startup.")
                continue
            try:
                nvidia_smi.nvmlDeviceGetClockInfo(self.gpu_handles[i],
                                                  nvidia_smi.NVML_CLOCK_GRAPHICS)
            except Exception as e:
                has_errors = True
                print(f"ERROR: GPU {i} has error: {e} ({type(e)}")
        return has_errors


def profile_ram():
    """

    Returns:
        RAM total (GB), RAM used (GB)
    """
    mem = psutil.virtual_memory()
    ram_total: float = mem.total / 1024 ** 3
    ram_used: float = mem.used / 1024 ** 3
    return ram_total, ram_used
