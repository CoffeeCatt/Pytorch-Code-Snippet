# To run argparse in jupyter notebook
import sys

parser = argparse.ArgumentParser(description='Pytorch Imagenet Training')
parser.add_argument('--data', type=str, default='~/datasets',help='location of the data corpus')

sys.argv = [''] #(added)
global args
args = parser.parse_args()
main_ssl(args)
