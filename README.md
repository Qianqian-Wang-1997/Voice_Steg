# Voice_Steg
Step 1： 准备数据 g729a有两个特征 LPC（三维）和PD（二维），用准备py脚本提取  
Step 2：运行cnn_lstm_att.py 训练数据为PD  
Step 3: 运行retrain_cnn_lstm_att.py 训练数据为LPC (testing)  
Step 4： 运行test.py (testing)  

