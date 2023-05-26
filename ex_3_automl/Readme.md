# AutoML exercises  - DL Lab 2023
In this task we will train a supernet using single-path NAS https://arxiv.org/abs/1904.00420
1. Your first task is to implement a sampler to randomly sample the one-hot encoded architecture (check ```optimizers/samplers/spos.py```) - 10 points
2. Your second task is to train the supernet using the ```train_spos.py``` for 1000 epochs. You will handover the trained model along with the completed code. - 10 points
3. Now we will run random-search on the pre-trained superent provided to you. Note that this pre-trained supernet is same as the one you trained in (3) except this one is trained for a higher training budget. Implement the 
```sample_random_config()``` method for the search space for random search. Run the random search for 100 epochs - 20 points
4. Now we will run evolutionary-search on the pre-trained superent provided to you. Complete the ```TODOS``` in the mutation and crossover blocks and run the evolutionary search for 20 epochs. - 20 points

# Final deliverables for the exercise are 
1. The complete code with the TODOs addressed
2. The pre-trained spos supernet trained for 1000 epochs
3. The incumbent trajectory for Random search. Report the validated error upon convergence
4. Saved checkpoint from the evolutionary search