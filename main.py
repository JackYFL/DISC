import argparse
from utils import load_config, get_log_name, set_seed, save_results, \
                  get_test_acc, print_config
from datasets import cifar_dataloader, clothing_dataloader, webvision_dataloader, food101N_dataloader, animal10N_dataloader, tiny_imagenet_dataloader
import algorithms
import numpy as np
import nni
import time

parser = argparse.ArgumentParser()
parser.add_argument('--config',
                    '-c',
                    type=str,
                    default='./configs/colearning.py',
                    help='The path of config file.')
parser.add_argument('--model_name', type=str, default='')
parser.add_argument('--dataset', type=str, default='cifar-10')
parser.add_argument('--root', type=str, default='/data/yfli/CIFAR10')
parser.add_argument('--save_path', type=str, default='./log/')
parser.add_argument('--gpu', type=str, default='1')
parser.add_argument('--noise_type', type=str, default='sym')
parser.add_argument('--percent', type=float, default=0.2)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--num_classes', type=int, default=100)
parser.add_argument('--momentum', type=float, default=0.99)
args = parser.parse_args()


def main():
    tuner_params = nni.get_next_parameter()
    config = load_config(args.config, _print=False)
    config.update(tuner_params)
    config['dataset'] = args.dataset
    config['root'] = args.root
    config['gpu'] = args.gpu
    config['noise_type'] = args.noise_type
    config['percent'] = args.percent
    config['seed'] = args.seed
    config['num_classes'] = args.num_classes
    config['momentum'] = args.momentum
    print_config(config)
    set_seed(config['seed'])

    if config['algorithm'] == 'DISC':
        model = algorithms.DISC(config,
                                input_channel=config['input_channel'],
                                num_classes=config['num_classes'])
        train_mode = 'train_index'
        
    elif config['algorithm'] == 'colearning':
        model = algorithms.Colearning(config,
                                      input_channel=config['input_channel'],
                                      num_classes=config['num_classes'])
        train_mode = 'train'
        
    elif config['algorithm'] == 'JointOptimization':
        model = algorithms.JointOptimization(
            config,
            input_channel=config['input_channel'],
            num_classes=config['num_classes'])
        train_mode = 'train_index'
        
    elif config['algorithm'] == 'GJS':
        model = algorithms.GJS(config,
                               input_channel=config['input_channel'],
                               num_classes=config['num_classes'])
        train_mode = 'train_index'
        
    elif config['algorithm'] == 'ELR':
        model = algorithms.ELR(config,
                               input_channel=config['input_channel'],
                               num_classes=config['num_classes'])
        train_mode = 'train_index'
        
    elif config['algorithm'] == 'PENCIL':
        model = algorithms.PENCIL(config,
                                  input_channel=config['input_channel'],
                                  num_classes=config['num_classes'])
        train_mode = 'train_index'
        
    else:
        model = algorithms.__dict__[config['algorithm']](
            config,
            input_channel=config['input_channel'],
            num_classes=config['num_classes'])
        train_mode = 'train_single'
        if config['algorithm'] == 'StandardCETest':
            train_mode = 'train_index'

    if 'cifar' in config['dataset']:
        dataloaders = cifar_dataloader(cifar_type=config['dataset'],
                                       root=config['root'],
                                       batch_size=config['batch_size'],
                                       num_workers=config['num_workers'],
                                       noise_type=config['noise_type'],
                                       percent=config['percent'])
        trainloader, testloader = dataloaders.run(
            mode=train_mode), dataloaders.run(mode='test')

    elif 'tiny_imagenet' in config['dataset']:
        tiny_imagenet_dataloaders = tiny_imagenet_dataloader(
            root_dir=config['root'],
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            noise_type=config['noise_type'],
            percent=config['percent'])
        trainloader, testloader = tiny_imagenet_dataloaders.run(
            mode=train_mode), tiny_imagenet_dataloaders.run(mode='test')

    elif 'clothing' in config['dataset']:
        clothing_dataloaders = clothing_dataloader(
            root_dir=config['root'],
            batch_size=config['batch_size'],
            num_workers=config['num_workers'])
        trainloader, evalloader, testloader = clothing_dataloaders.run()

    elif 'webvision' in config['dataset']:
        webvision_dataloaders = webvision_dataloader(
            root_dir_web=config['root'],
            root_dir_imgnet=config['imgnet_root'],
            batch_size=config['batch_size'],
            num_workers=config['num_workers'])
        trainloader, evalloader, testloader = webvision_dataloaders.run()

    elif 'food101N' in config['dataset']:
        food101N_dataloaders = food101N_dataloader(
            root_dir=config['root'],
            batch_size=config['batch_size'],
            num_workers=config['num_workers'])
        trainloader, testloader = food101N_dataloaders.run(mode=train_mode)

    elif 'animal' in config['dataset']:
        animal10N_dataloaders = animal10N_dataloader(
            root_dir=config['root'],
            batch_size=config['batch_size'],
            num_workers=config['num_workers'])
        trainloader, testloader = animal10N_dataloaders.run(mode=train_mode)

    num_test_images = len(testloader.dataset)

    start_epoch = 0
    epoch = 0

    # evaluate models with random weights
    if 'webvision' in config['dataset']:
        test_acc = model.evaluate(testloader)
        print(
            'Epoch [%d/%d] Test Accuracy on the %s test images: top1: %.4f, top5: %.4f'
            % (epoch, config['epochs'], num_test_images, test_acc[0],
               test_acc[1]))
    else:
        test_acc = get_test_acc(model.evaluate(testloader))
        print('Epoch [%d/%d] Test Accuracy on the %s test images: %.4f' %
              (epoch, config['epochs'], num_test_images, test_acc))

    acc_list, acc_all_list = [], []
    best_acc, best_epoch = 0.0, 0

    if config['algorithm'] == 'DISC' or config['algorithm'] == 'StandardCETest':
        if 'cifar' in config['dataset']:
            model.get_labels(trainloader)
        elif 'clothing' in config['dataset']:
            model.get_clothing_labels(config['root'])
            start_epoch = 0
        elif 'webvision' in config['dataset']:
            model.get_webvision_labels(config['root'])
        elif 'food101N' in config['dataset']:
            model.get_food101N_labels(config['root'])
        elif 'animal' in config['dataset']:
            model.get_animal10N_labels(config['root'])
        model.weak_labels = model.labels.detach().clone()
        print('The labels are loaded!!!')

    since = time.time()
    for epoch in range(start_epoch, config['epochs']):
        # train
        model.train(trainloader, epoch)

        if 'webvision' in config['dataset']: # webvision needs to validate on webvision's val set and ImageNet's test set
            val_acc = model.evaluate(evalloader)
            print(
                'Epoch [%d/%d] Val Accuracy on the %s val images: top1: %.4f top5: %.4f %%'
                % (epoch + 1, config['epochs'], num_test_images, val_acc[0],
                   val_acc[1]))

            test_acc = model.evaluate(testloader)
            if best_acc < test_acc[0]:
                best_acc, best_epoch = test_acc[0], epoch

            print(
                'Epoch [%d/%d] Test Accuracy on the %s test images: top1: %.4f, top5: %.4f. %%'
                % (epoch + 1, config['epochs'], num_test_images, test_acc[0],
                   test_acc[1]))

            if epoch >= config['epochs'] - 10:
                acc_list.extend([test_acc[0]])
            acc_all_list.extend([test_acc[0]])

        else:
            # evaluate
            test_acc = get_test_acc(model.evaluate(testloader))

            if best_acc < test_acc:
                best_acc, best_epoch = test_acc, epoch

            print(
                'Epoch [%d/%d] Test Accuracy on the %s test images: %.4f %%' %
                (epoch + 1, config['epochs'], num_test_images, test_acc))

            if epoch >= config['epochs'] - 10:
                acc_list.extend([test_acc])

            if 'clothing' in config['dataset']:
                trainloader, evalloader, testloader = clothing_dataloaders.run(
                )
            acc_all_list.extend([test_acc])

    time_elapsed = time.time() - since
    total_min = time_elapsed // 60
    hour = total_min // 60
    min = total_min % 60
    sec = time_elapsed % 60

    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        hour, min, sec))

    if config['save_result']:
        config['algorithm'] = config['algorithm'] + args.model_name
        acc_np = np.array(acc_list)
        nni.report_final_result(acc_np.mean())
        jsonfile = get_log_name(config, path=args.save_path)
        np.save(jsonfile.replace('.json', '.npy'), np.array(acc_all_list))
        save_results(config=config,
                     last_ten=acc_np,
                     best_acc=best_acc,
                     best_epoch=best_epoch,
                     jsonfile=jsonfile)


if __name__ == '__main__':
    main()