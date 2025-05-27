import argparse
import torch
from torchvision import transforms

from retinanet import model
from retinanet.dataloader import CocoDataset, Resizer, Normalizer
from retinanet import coco_eval

assert torch.__version__.split('.')[0] == '2'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--coco_path', help='Path to COCO directory')
    parser.add_argument('--model_path', help='Path to model', type=str)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser = parser.parse_args(args)

    dataset_val = CocoDataset(parser.coco_path, set_name='val2017',
                              transform=transforms.Compose([Normalizer(), Resizer()]))
    checkpoint = torch.load(parser.resume, map_location='cpu')
    # Create the model
    retinanet = model.resnet50(num_classes=dataset_val.num_classes(), pretrained=True,resume=checkpoint)

    use_gpu = True

    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()

    if torch.cuda.is_available():
        #retinanet.load_state_dict(torch.load(parser.model_path))
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        #retinanet.load_state_dict(torch.load(parser.model_path))
        retinanet = torch.nn.DataParallel(retinanet)
    
    print('Loaded weights from {}'.format(parser.model_path))
    retinanet.training = False
    retinanet.eval()
    retinanet.module.freeze_bn()
    print('Evaluating dataset')
    coco_eval.evaluate_coco(dataset_val, retinanet,threshold=0.0)

    
    print('Evaluation done')


if __name__ == '__main__':
    main()
