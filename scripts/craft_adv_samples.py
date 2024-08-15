import os
from util import get_dataloader, get_imagenet_classifier, get_cifar_classifier, set_seed, evaluate, get_adv_loader, plot_images, calc_Freq, get_fft_feature, local_histogram_equalization
import torch
import torch.nn as nn
import torchattacks
from tqdm import tqdm
import global_variable


base_dir = global_variable.base_dir


ATTACK_PARAMS = {
    'mnist': {'eps': 0.300, 'eps_iter': 0.010},
    'cifar': {'eps': 0.050, 'eps_iter': 0.005},
    'svhn': {'eps': 0.130, 'eps_iter': 0.010},
    'imagenet': {'eps': 8, 'eps_iter': 0.010}
}


def craft_one_type(classifier, loader, dataset, attack, eps, batch_size, criterion, device, one_three, classifier_name):
    print(f'Crafting {attack}(eps is: {eps}/255) adversarial samples...')
    if attack == 'FGSM':
        atk = torchattacks.FGSM(classifier, eps=eps/255)
    elif attack == "BIM":
        atk = torchattacks.BIM(classifier, eps=eps/255, steps=10)
    elif attack == 'PGD':
        atk = torchattacks.PGD(classifier, eps=eps / 255, alpha=0.01, steps=10)
    elif attack == 'APGD':
        atk = torchattacks.APGD(classifier, eps=eps / 255, steps=10)
    elif attack == 'PGDRS':
        atk = torchattacks.PGDRS(classifier, eps=eps / 255, steps=10)
    elif attack == 'DIFGSM':
        atk = torchattacks.DIFGSM(classifier, eps=eps / 255, steps=10)
    elif attack == 'MIFGSM':
        atk = torchattacks.MIFGSM(classifier, eps=eps / 255, steps=10)
    elif attack == 'DeepFool':
        atk = torchattacks.DeepFool(classifier, steps=50)
    elif attack == 'CW':
        # TODO: CW attack
        atk = torchattacks.CW(classifier, c=1, kappa=0, steps=500, lr=0.01)

    else:
        pass
        # JSMA attack
        # print('Crafting jsma adversarial samples. This may take a while...')
        # X_adv = saliency_map_method(
        #     sess, model, X, Y, theta=1, gamma=0.1, clip_min=0., clip_max=1.
        # )
    
    i = 0
    classifier.eval()
    true = []
    pred = []
    if not os.path.isdir(f'{base_dir}/adv_data/{dataset}/{classifier_name}/{one_three}/{attack}'):
        os.makedirs(f'{base_dir}/adv_data/{dataset}/{classifier_name}/{one_three}/{attack}')
    adv_x, adv_y = [], []
    for (img, label) in tqdm(loader):
        img = img.to(device)
        label = label.to(device)
        X_adv = atk(img, label)
        outputs = classifier(X_adv)
        pred.append(outputs.argmax(dim=1))
        true.append(label)
        # adv_x.append(X_adv)
        # adv_y.append(label)
        adv_x.append(X_adv.cpu())
        adv_y.append(label.cpu())
    # adv_x = torch.cat(adv_x, 0)
    # adv_y = torch.cat(adv_y, 0)
    # adv_imgs = {'X': adv_x.cpu(), 'Y': adv_y.cpu()}
    adv_imgs = {'X': torch.cat(adv_x, 0), 'Y': torch.cat(adv_y, 0)}
    torch.save(adv_imgs, f'{base_dir}/adv_data/{dataset}/{classifier_name}/{one_three}/{attack}/eps_{eps}.pkl')

    true = torch.cat(true, dim=0)
    pred = torch.cat(pred, dim=0)
    correct_predictions = pred.eq(true).sum()
    accuracy = correct_predictions / len(loader.dataset) * 100

    print("Model accuracy on the %s(eps is: %d/255) adversarial test set: %0.2f%%" %
          (args.attack, eps, accuracy.cpu().numpy()))

    # Load adversarial samples
    # test_adv_loader = get_adv_loader(args, args.dataset, args.attack, one_three, net_name, eps)
    #
    # for s_type, ld in zip(['normal', 'adversarial'], [loader, test_adv_loader]):
    #     if s_type == 'normal':
    #         _, acc, pred, true = evaluate(model, ld, criterion, device, return_pred=True)
    #         print("Model accuracy on the %s test set: %0.2f%%" % (s_type, acc))
    #     else:
    #         _, acc = evaluate(model, ld, criterion, device)
    #         print("Model accuracy on the %s(eps is: %d/255) %s test set: %0.2f%%" % (args.attack, eps, s_type, acc))

    
def main(args):
    set_seed(0)
    assert args.dataset in ['mnist', 'cifar', 'svhn', 'imagenet'], \
        "Dataset parameter must be either 'mnist', 'cifar', 'svhn' or 'imagenet'"
    assert args.attack in ['FGSM', 'BIM', 'PGD', 'TPGD', 'APGD', 'PGDRS', 'DIFGSM', 'MIFGSM', 'DeepFool', 'CW'], \
        "Attack parameter must be either 'fgsm', 'pgd', 'bim-a', 'bim-b', " \
        "'jsma' or 'cw'"
    assert os.path.isfile(f'{base_dir}/checkpoint/{args.dataset}/classifier_{args.classifier}.pth'), \
        'model file not found... must first train model using train_model.py.'
    print('Dataset: %s.         Attack: %s' % (args.dataset, args.attack))
    
    train_loader, valid_loader, test_loader = get_dataloader(args)
    criterion = nn.CrossEntropyLoss()
    if args.dataset == 'imagenet':
        classifier, classifier_name = get_imagenet_classifier(args.classifier, pretrained=False)
    elif args.dataset == 'cifar100':
        classifier, classifier_name = get_cifar_classifier(args.classifier, num_classes=100, pretrained=False)
    else:
        classifier, classifier_name = get_cifar_classifier(args.classifier, num_classes=10, pretrained=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Used device : {device};          model name: {classifier_name}')
    classifier.to(device)
    weight = torch.load(f'{base_dir}/checkpoint/{args.dataset}/classifier_{classifier_name}.pth')
    classifier.load_state_dict(weight, strict=False)
    train_loss, train_acc = evaluate(classifier, train_loader, criterion, device)
    print('test acc : {:.2f}% \t test loss : {:.4f}'.format(train_acc, train_loss))

    if args.attack == 'all':
        # Cycle through all attacks
        for attack in ['fgsm', 'bim-a', 'bim-b', 'jsma', 'cw']:
            craft_one_type(classifier, test_loader, args.dataset, attack,
                           args.batch_size, criterion, device, classifier_name)
    elif args.attack == 'DeepFool':
        for loader_type, loader_data in zip(['train_loader', 'valid_loader', 'test_loader'],
                                            [train_loader, valid_loader, test_loader]):
            craft_one_type(classifier, loader_data, args.dataset, args.attack, 12,
                           args.batch_size, criterion, device, loader_type, classifier_name)
            print(f'Base {classifier_name} Adversarial samples({loader_type})'
                  f'dataset len:{len(loader_data.dataset)}')
    elif args.attack == 'CW':
        for loader_type, loader_data in zip(['train_loader', 'valid_loader', 'test_loader'],
                                            [train_loader, valid_loader, test_loader]):
            craft_one_type(classifier, loader_data, args.dataset, args.attack, 12,
                           args.batch_size, criterion, device, loader_type, classifier_name)
            print(f'Base {classifier_name} Adversarial samples({loader_type})'
                  f'dataset len:{len(loader_data.dataset)}')
    else:
        # Craft one specific attack type
        for eps in [4]:
            for loader_type, loader_data in zip(['train_loader', 'valid_loader', 'test_loader'],
                                                [train_loader, valid_loader, test_loader]):
                craft_one_type(classifier, loader_data, args.dataset, args.attack, eps,
                               args.batch_size, criterion, device, loader_type, classifier_name)
            print(f'Base {classifier_name} Adversarial samples({loader_type}) esp: {eps} crafted and saved,'
                  f'dataset len:{len(loader_data.dataset)}')
        
    print(f'Adversarial samples crafted and saved')


class Args():
    dataset = 'cifar'
    classifier = 'vgg19'
    attack = 'DIFGSM'
    batch_size = 64


    
if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     '-m', '--model',
    #     help="classifier model; torch_nets in net",
    #     required=True, type=str
    # )
    # parser.add_argument(
    #     '-d', '--dataset',
    #     help="Dataset to use; either 'mnist', 'cifar' or 'svhn'",
    #     required=True, type=str
    # )
    # parser.add_argument(
    #     '-a', '--attack',
    #     help="Attack to use; either 'fgsm', 'bim-a', 'bim-b', 'jsma', 'cw' "
    #          "or 'all'",
    #     required=True, type=str
    # )
    # parser.add_argument(
    #     '-b', '--batch_size',
    #     help="The batch size to use for training.",
    #     required=False, type=int
    # )
    # parser.set_defaults(batch_size=256)
    # args = parser.parse_args()
    args =Args

    main(args)