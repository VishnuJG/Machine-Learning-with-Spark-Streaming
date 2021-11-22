#! /usr/bin/python3

import time
import json
import pickle
import socket
import argparse
import numpy as np
import pandas as pd
from tqdm.auto import tqdm


parser=argparse.ArgumentParser(description="Streaming a file to a spark streaming context")
parser.add_argument('--file','-f',help='File to stream',required=False, type=str, default='cifar')
parser.add_argument('--batch-size','-b',help='Batch size',required=False, type=int, default=100)
parser.add_argument('--endless','-e',help='Enable endless streaming',required=False, type=bool, default=False)


TCP_IP="localhost"
TCP_PORT=6100

def connectTCP():
    s=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR,1)
    s.bind((TCP_IP,TCP_PORT))
    s.listen(1)
    print(f"waiting for connection on port {TCP_PORT}...")
    connection, address=s.accept()
    print(f"Connected to {address}")
    return connection, address

def sendCIFARBatchFileToSpark(tcp_connection, input_batch_file):
    with open(f'cifar/{input_batch_file}','rb') as batch_file:
        batch_data=pickle.load(batch_file, encoding='bytes')

    data=batch_data[b'data']
    data=list(map(np.ndarray.tolist,data))
    labels=batch_data[b'labels']
    feature_size=len(data[0])
    for image_index in tqdm(range(0,len(data)-batch_size+1, batch_size)):
        image_data_batch=data[image_index:image_index+batch_size]
        image_label=labels[image_index:image_index+batch_size]

        payload=dict()
        for mini_batch_index in range(len(image_data_batch)):
            payload[mini_batch_index]= dict()
            for feature_index in range(feature_size):
                payload[mini_batch_index][f'feature{feature_index}']=image_data_batch[mini_batch_index][feature_index]
            payload[mini_batch_index]['label']=image_label[mini_batch_index]
        send_batch=(json.dumps(payload)+'\n').encode()

        try:
            tcp_connection.send(send_batch)
        except BrokenPipeError:
            print("Either batch size is too big for the dataset or the connection was closed")
        except Exception as error_message:
            print(f"Exception thrown but was handled: {error_message}")
        time.sleep(1)


def streamCIFARDataset(tcp_connection, dataset_type='cifar'):
    print("starting to stream CIFAR data")
    CIFAR_BATCHES=[
        'data_batch_1',
        # 'data_batch_2',   # uncomment to stream the second training dataset
        # 'data_batch_3',   # uncomment to stream the third training dataset
        # 'data_batch_4',   # uncomment to stream the fourth training dataset
        # 'data_batch_5',    # uncomment to stream the fifth training dataset
        # 'test_batch'      # uncomment to stream the test dataset
    ]
    for batch in CIFAR_BATCHES:
        sendCIFARBatchFileToSpark(tcp_connection,batch)
        time.sleep(1)

if __name__ == '__main__':
    args=parser.parse_args()
    print(args)
    input_file=args.file
    batch_size=args.batch_size
    endless=args.endless

    tcp_connection,_=connectTCP()


    if endless:
        while True:
            streamCIFARDataset(tcp_connection, input_file)
    else:
        streamCIFARDataset(tcp_connection,input_file)


