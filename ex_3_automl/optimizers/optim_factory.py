
from optimizers.mixop.spos import SPOSMixOp
from optimizers.sampler.spos import SPOSSampler


def get_mixop(opt_name, use_we_v2=False):
    if not use_we_v2:
        if opt_name == "spos":
            return SPOSMixOp()
        else:
            raise NotImplementedError(f'WE v2 is not implemented for {opt_name}')

def get_sampler(opt_name):
    if opt_name == "spos":
        return SPOSSampler()
    else:
        raise NotImplementedError(f'Sampler is not implemented for {opt_name}')
