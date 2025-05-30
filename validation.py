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

    parser = parser.parse_args(args)

    dataset_val = CocoDataset(parser.coco_path, set_name='test2017',
                              transform=transforms.Compose([Normalizer(), Resizer()]))

    # Create the model
    retinanet = torch.load(parser.model_path)
    base_model = retinanet.module if isinstance(retinanet, torch.nn.DataParallel) else retinanet
    use_gpu = True
    # 3. Move to GPU
    if torch.cuda.is_available():
        base_model = base_model.cuda()

    # 4. Freeze BNs on the raw model
    base_model.freeze_bn()

    # 5. (Re-)wrap for multi-GPU, if you want
    retinanet = torch.nn.DataParallel(base_model) if torch.cuda.device_count()>1 else base_model


    retinanet.training = False
    retinanet.eval()
    
    
    print('Evaluating COCO dataset...')
    stats, average_time = coco_eval.evaluate_coco(dataset_val, retinanet)
    print('Average time: {:.1f} ms'.format(average_time*1000))


if __name__ == '__main__':
    main()