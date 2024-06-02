# !/usr/bin/env python3


batch_size = 256

cic_feature_num = 58
cic_hiddens1 = 80
cic_hiddens2 = 64
cic_output1 = 32
cic_output2 = 9

# cic_normal_size = 32
# cic_ginput_size = cic_output2 + cic_normal_size
# cic_ghiddens1 = 64
# cic_ghiddens2 = 96

label_names = ['Benign', 'FTP-Patator', 'SSH-Patator', 'DoS-GoldenEye',
          'DoS-Hulk', 'DoS-Slowloris', 'DoS-SlowHttpTest', 'Bot', 'Web-BruteForce']
