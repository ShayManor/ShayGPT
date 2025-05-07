import math

s = """
epoch 1 step 14600 loss 4.0130 lr = 6.84e-05 time = 0.43373966217041016
epoch 1 step 14700 loss 4.0450 lr = 6.885e-05 time = 0.43677735328674316
epoch 1 step 14800 loss 4.0719 lr = 6.9375e-05 time = 0.35410380363464355
epoch 1 step 14900 loss 4.0358 lr = 6.9825e-05 time = 0.43401074409484863
epoch 1 step 15000 loss 4.0474 lr = 7.0275e-05 time = 0.434603214263916
epoch 1 step 15100 loss 4.1407 lr = 7.0725e-05 time = 0.43564438819885254
epoch 1 step 15200 loss 4.1804 lr = 7.125e-05 time = 0.3518564701080322
epoch 1 step 15300 loss 4.1792 lr = 7.17e-05 time = 0.4334073066711426
epoch 1 step 15400 loss 4.1440 lr = 7.215e-05 time = 0.43529462814331055
epoch 1 step 15500 loss 4.1183 lr = 7.26e-05 time = 0.43607425689697266
epoch 1 step 15600 loss 4.0697 lr = 7.3125e-05 time = 0.35144925117492676
epoch 1 step 15700 loss 4.0554 lr = 7.3575e-05 time = 0.43399858474731445
epoch 1 step 15800 loss 3.9775 lr = 7.4025e-05 time = 0.4346308708190918
epoch 1 step 15900 loss 3.9560 lr = 7.4475e-05 time = 0.43506288528442383
epoch 1 step 16000 loss 4.0027 lr = 7.5e-05 time = 0.3509664535522461
epoch 1 step 16100 loss 3.9984 lr = 7.545e-05 time = 0.4346606731414795
epoch 1 step 16200 loss 4.0168 lr = 7.59e-05 time = 0.4340641498565674
epoch 1 step 16300 loss 3.9686 lr = 7.635e-05 time = 0.43388962745666504
epoch 1 step 16400 loss 4.0385 lr = 7.6875e-05 time = 0.35112547874450684
epoch 1 step 16500 loss 3.9760 lr = 7.7325e-05 time = 0.43387722969055176
epoch 1 step 16600 loss 3.8975 lr = 7.7775e-05 time = 0.4358196258544922
"""
tot = 0
counter = 0
for idx, line in enumerate(s.split('loss ')):
    if idx == 0:
        continue
    tot += float(line.split(' ')[0])
    counter += 1
print(f'Total: {tot}, {tot / counter}, perplexity: {math.e ** (tot / counter)}')