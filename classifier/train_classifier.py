"""
Entry point: python scripts/train_classifier.py --data data/tfrecords/
"""
import argparse
from classifier.model import train

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',   default='data/tfrecords/')
    parser.add_argument('--output', default='classifier/model.tflite')
    args = parser.parse_args()
    train(args.data, args.output)