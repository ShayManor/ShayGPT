import math

s = """
epoch 10 step 100500 loss 2.4449 lr = 9.4215e-05 time = 0.45780467987060547
epoch 10 step 100600 loss 2.4727 lr = 9.4305e-05 time = 0.4575462341308594
epoch 10 step 100700 loss 2.4961 lr = 9.4395e-05 time = 0.45692873001098633
epoch 10 step 100800 loss 2.4762 lr = 9.45e-05 time = 0.37107229232788086
epoch 10 step 100900 loss 2.4924 lr = 9.459e-05 time = 0.4548788070678711
epoch 10 step 101000 loss 2.4986 lr = 9.468e-05 time = 0.454268217086792
epoch 10 step 101100 loss 2.5032 lr = 9.477e-05 time = 0.456451416015625
epoch 10 step 101200 loss 2.5112 lr = 9.4875e-05 time = 0.3702430725097656
epoch 10 step 101300 loss 2.5523 lr = 9.4965e-05 time = 0.453449010848999
epoch 10 step 101400 loss 2.5844 lr = 9.5055e-05 time = 0.4547719955444336
epoch 10 step 101500 loss 2.5610 lr = 9.5145e-05 time = 0.4550788402557373
epoch 10 step 101600 loss 2.5669 lr = 9.525e-05 time = 0.3718595504760742
epoch 10 step 101700 loss 2.5885 lr = 9.534e-05 time = 0.45407676696777344
epoch 10 step 101800 loss 2.5612 lr = 9.543e-05 time = 0.4517970085144043
epoch 10 step 101900 loss 2.5182 lr = 9.552e-05 time = 0.4525308609008789
epoch 10 step 102000 loss 2.5050 lr = 9.5625e-05 time = 0.36719489097595215
epoch 10 step 102100 loss 2.5677 lr = 9.5715e-05 time = 0.4516589641571045
epoch 10 step 102200 loss 2.5236 lr = 9.5805e-05 time = 0.45079731941223145
epoch 10 step 102300 loss 2.4788 lr = 9.5895e-05 time = 0.45220947265625
epoch 10 step 102400 loss 2.4819 lr = 9.6e-05 time = 0.367018461227417
epoch 10 step 102500 loss 2.4883 lr = 9.609e-05 time = 0.45041823387145996
"""
tot = 0
counter = 0
for idx, line in enumerate(s.split('loss ')):
    if idx == 0:
        continue
    tot += float(line.split(' ')[0])
    counter += 1
print(f'Total: {tot}, {tot / counter}, perplexity: {math.e ** (tot / counter)}')