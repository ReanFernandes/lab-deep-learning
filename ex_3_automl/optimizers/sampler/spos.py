from optimizers.sampler.base_sampler import Sampler
import torch
import torch.nn.functional as F
import numpy as np


class SPOSSampler(Sampler):

    def sample_epoch(self, alphas_list):
        sampled_alphas_list = []
        for alpha in alphas_list:
            sampled_alphas_list.append(self.sample(alpha ))
        return sampled_alphas_list

    def sample_step(self, alphas_list):
        sampled_alphas_list = []
        for alpha in alphas_list:
            sampled_alphas_list.append(self.sample(alpha))
        return sampled_alphas_list

    def sample_indices(self, num_steps, num_selected):
        indices_to_sample = []
        start = 0
        n = 2
        while True:
            end = start+n
            if end > num_steps:
                break
            choices = np.random.choice(
                [i for i in range(start, end)], num_selected, replace=False)
            for c in list(choices):
                indices_to_sample.append(c)
            start = end
            n = n+1
        return indices_to_sample

    def sample(self, alpha):
        '''
        TODO: for alpha of any shape return an alpha one-hot encoded along the last dimension 
        (i.e. the dimension of the choices)
        Example 1 alpha = [-0.1, 0.2, -0.3, 0.4] -> Sample any index from 0 to 3 and return a one-hot encoded vector eg: [0, 0, 1, 0]
        Example 2 alpha = [[-0.1, 0.2, -0.3, 0.4], [-0.1, 0.2, -0.3, 0.4]] -> Sample any index from 0 to 3 and return a one-hot encoded vector eg: [[0, 0, 1, 0], [1, , 0, 0]]
        Args:
            alpha (torch.Tensor): alpha values of any shape
            Returns: torch.Tensor: one-hot encoded tensor of the same shape as alpha
        '''

        #ensure that alpha is a tensor
        if not torch.is_tensor(alpha):
            alpha = torch.tensor(alpha)
        
        num_classes = alpha.size()[-1]
        shape = alpha.shape[:-1]  # Shape of all dims except last 
        
        # Sample random indices   
        idx = torch.randint(0, num_classes, shape)
        
        # One-hot encode samples
        one_hot = F.one_hot(idx, num_classes=num_classes)
        
        # Check if one_hot has the correct shape 
        if one_hot.shape == alpha.shape:  
            # If so, we're done! Return one_hot
            return one_hot 
        else:
            # Recurse and try again
            return self.sample(alpha)
            


# test spos
if __name__ == '__main__':
    alphas = torch.randn([14,8])
    sampler = SPOSSampler()
    sampled_alphas = sampler.sample(alphas)
    print(sampled_alphas)
