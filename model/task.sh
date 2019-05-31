#!/bin/bash
echo run p2p_model!

cd /home/nzhang/p2p_serve && nohup /root/anaconda3/envs/py36/bin/python3.6 /home/nzhang/p2p_serve/p2p_ensemble_oss.py &
