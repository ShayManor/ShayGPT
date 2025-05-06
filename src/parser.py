import math

s = """
epoch 0 step 234000 loss 3.4260 lr = 2.4073e-07
epoch 0 step 234100 loss 3.4255 lr = 2.3309e-07
epoch 0 step 234200 loss 3.1428 lr = 2.2495e-07
epoch 0 step 234300 loss 3.7958 lr = 2.1757e-07
epoch 0 step 234400 loss 3.5480 lr = 2.0972e-07
epoch 0 step 234500 loss 3.2858 lr = 2.0259e-07
epoch 0 step 234600 loss 3.4518 lr = 1.9501e-07
epoch 0 step 234700 loss 2.8406 lr = 1.8814e-07
epoch 0 step 234800 loss 3.3265 lr = 1.8084e-07
epoch 0 step 234900 loss 3.3641 lr = 1.7422e-07
epoch 0 step 235000 loss 3.0871 lr = 1.672e-07
epoch 0 step 235100 loss 3.6191 lr = 1.6084e-07
epoch 0 step 235200 loss 2.9102 lr = 1.541e-07
epoch 0 step 235300 loss 3.2131 lr = 1.48e-07
epoch 0 step 235400 loss 3.4298 lr = 1.4153e-07
epoch 0 step 235500 loss 3.7234 lr = 1.3568e-07
epoch 0 step 235600 loss 3.4581 lr = 1.2949e-07
epoch 0 step 235700 loss 3.5868 lr = 1.239e-07
epoch 0 step 235800 loss 3.3463 lr = 1.1799e-07
epoch 0 step 235900 loss 3.3715 lr = 1.1266e-07
epoch 0 step 236000 loss 3.4765 lr = 1.0702e-07
epoch 0 step 236100 loss 3.3774 lr = 1.0195e-07
"""
tot = 0
counter = 0
for idx, line in enumerate(s.split('loss ')):
    if idx == 0:
        continue
    tot += float(line.split(' ')[0])
    counter += 1
print(f'Total: {tot}, {tot / counter}, perplexity: {math.e ** (tot / counter)}')