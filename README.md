# yizhi

1、put the given data into /path/to/yizhi/data

data
    WOZ
        train.csv
        val.csv
        test.csv
dataloader
models
output
tools


2、running train by 
sh train.sh



*****************report***************

采用要求的模型训练，去除stopwords使用bert分词，30轮可收敛，验证f1_macro:0.7886935960841901, f1_micro:0.9342634192480492, loss:0.0007

测试2024-03-26 19:51:57,219   INFO  f1_macro:0.7855377539597839, f1_micro:0.9424256724870221, loss:0.0007
观察测试数据集  其中有'''thank	你好，我想吃美食街，帮我推荐一个人均消费在50-100元的餐馆，谢谢。''''对测试效果有一定影响

