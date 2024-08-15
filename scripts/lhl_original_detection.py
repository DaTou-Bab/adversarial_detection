import os
from his_equ_net import his_equ_net
from sklearn.metrics import accuracy_score, precision_score, recall_score
from util import (get_dataloader, get_imagenet_classifier, get_cifar_classifier, set_seed, plot_images, calc_Freq,
                  compute_roc, get_adv_loader, evaluate, get_subsample_loader,
                  local_histogram_equalization
                  )

import torch
import torch.nn as nn
import torch.nn.functional as F
import global_variable
base_dir = global_variable.base_dir


def l2_penalty(w):
    return (w ** 2).sum() / 2


def main(args):
    set_seed(0)
    assert args.dataset in ['mnist', 'cifar', 'svhn', 'imagenet'], \
        "Dataset parameter must be either 'mnist', 'cifar' 'svhn' or 'imagenet'"
    assert args.attack in ['FGSM', 'BIM', 'PGD', 'APGD', 'PGDRS', 'DIFGSM', 'MIFGSM', 'DeepFool', 'CW'], \
        "Attack parameter must be either 'fgsm', 'pgd', 'bim-a', 'bim-b', " \
        "'jsma' or 'cw'"
    assert os.path.isfile(f'{base_dir}/checkpoint/{args.dataset}/classifier_{args.classifier}.pth'), \
        'model file not found... must first train model using train_model.py.'
    assert os.path.isdir(f'{base_dir}/adv_data/{args.dataset}/{args.classifier}/train_loader/{args.attack}'), \
        'adversarial sample file not found... must first craft adversarial ' \
        'samples using craft_adv_samples.py'
    assert os.path.isdir(f'{base_dir}/adv_data/{args.dataset}/{args.classifier}/valid_loader/{args.attack}'), \
        'adversarial sample file not found... must first craft adversarial ' \
        'samples using craft_adv_samples.py'
    assert os.path.isdir(f'{base_dir}/adv_data/{args.dataset}/{args.classifier}/test_loader/{args.attack}'), \
        'adversarial sample file not found... must first craft adversarial ' \
        'samples using craft_adv_samples.py'
    print(f'Loading the data(data is {args.dataset}) and model(model is {args.classifier})...')

    # Load the model
    criterion = nn.CrossEntropyLoss()
    if args.dataset == 'imagenet':
        classifier, classifier_name = get_imagenet_classifier(args.classifier, pretrained=False)
    elif args.dataset == 'cifar100':
        classifier, classifier_name = get_cifar_classifier(args.classifier, dataset_num=100, pretrained=False)
    else:
        classifier, classifier_name = get_cifar_classifier(args.classifier, num_classes=10, pretrained=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Used device : {device}')
    classifier.to(device)
    weight = torch.load(f'{base_dir}/checkpoint/{args.dataset}/classifier_{classifier_name}.pth')
    classifier.load_state_dict(weight, strict=False)

    # Load the dataset
    train_loader, valid_loader, test_loader = get_dataloader(args)
    eps = args.epsilon
    # Check attack type, select adversarial and noisy samples accordingly
    print(f'Loading {args.attack}(esp is: {eps}) adversarial samples...')
    if args.attack == 'all':
        # TODO: implement 'all' option
        # X_test_adv = ...
        # X_test_noisy = ...
        raise NotImplementedError("'All' types detector not yet implemented.")
    else:
        # Load adversarial samples
        train_adv_loader = get_adv_loader(args, args.dataset, args.attack, 'train_loader', classifier_name, eps)
        valid_adv_loader = get_adv_loader(args, args.dataset, args.attack, "valid_loader", classifier_name, eps)
        test_adv_loader = get_adv_loader(args, args.dataset, args.attack, 'test_loader', classifier_name, eps)

    # Check model accuracies on each sample type
    for s_type, loader in zip(['train_normal', 'train_adversarial', 'valid_normal', 'valid_adversarial'],
                              [train_loader, train_adv_loader, valid_loader, valid_adv_loader]):
        if s_type == 'train_normal':
            _, acc, train_pred, train_true = evaluate(classifier, loader, criterion, device, return_pred=True)
            print("Model accuracy on the %s test set: %0.2f%%" % (s_type, acc))
        elif s_type == 'train_adversarial':
            _, acc = evaluate(classifier, loader, criterion, device)
            print("Model accuracy on the %s test set: %0.2f%%" % (s_type, acc))
        elif s_type == 'valid_normal':
            _, acc, valid_pred, valid_true = evaluate(classifier, loader, criterion, device, return_pred=True)
            print("Model accuracy on the %s test set: %0.2f%%" % (s_type, acc))
        else:
            _, acc = evaluate(classifier, loader, criterion, device)
            print("Model accuracy on the %s test set: %0.2f%%" % (s_type, acc))

    train_inds_correct = torch.where((train_true == train_pred))[0]
    train_loader = get_subsample_loader(args, train_loader, train_inds_correct)
    train_adv_loader = get_subsample_loader(args, train_adv_loader, train_inds_correct)

    valid_inds_correct = torch.where((valid_true == valid_pred))[0]
    valid_loader = get_subsample_loader(args, valid_loader, valid_inds_correct)
    valid_adv_loader = get_subsample_loader(args, valid_adv_loader, valid_inds_correct)

    EPOCHS = args.epochs
    lr = 1e-4
    detection_net = his_equ_net()
    detection_net_name = 'lhl'
    optimizer = torch.optim.Adam(detection_net.parameters(), lr=lr)
    criterion_loss = nn.CrossEntropyLoss()
    detection_net.to(device)

    best_v_loss = 10000.0
    best_accuracy = 0.0
    losses = []
    for epoch in range(EPOCHS):
        running_loss = 0.0
        detection_net.train()
        for i, (x_original, x_adv) in enumerate(zip(train_loader, train_adv_loader)):
            x, x_adv = x_original[0].to(device), x_adv[0].to(device)
            labels = torch.cat((torch.zeros(x.shape[0], dtype=torch.long),
                                torch.ones(x_adv.shape[0], dtype=torch.long))).to(device)
            x_input = torch.cat((x, x_adv), dim=0).to(device)
            lhe_image = local_histogram_equalization(x_input, 8).to(device)

            x_input_4low = F.interpolate(
                F.interpolate(lhe_image, scale_factor=1 / 2, mode='bicubic', align_corners=False,
                              recompute_scale_factor=True), scale_factor=2, mode='bicubic', align_corners=False,
                recompute_scale_factor=True).to(device) # 4倍high-pass

            x_input_4high = lhe_image - x_input_4low
            x_input_4high.to(device)
            x_input_4low = torch.autograd.Variable(x_input_4low, requires_grad=True)

            optimizer.zero_grad()
            y_logit = classifier(x_input_4low)
            pred_class = y_logit.argmax(dim=1)
            loss = criterion(y_logit, pred_class)
            gradient = torch.autograd.grad(loss, x_input_4low)[0]
            gradient = torch.abs(gradient).detach().cuda()

            mini_out = detection_net(x_input_4high, gradient)
            mini_loss = criterion_loss(mini_out, labels.long())
            mini_loss.backward()
            optimizer.step()
            running_loss += mini_loss.item()

            if (i + 1) % 20 == 0:  # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 20:.3f}')
                running_loss = 0.0

        if not os.path.isdir(f'{base_dir}/checkpoint/{args.dataset}/{detection_net_name}/{classifier_name}/{args.attack}'):
            os.makedirs(f'{base_dir}/checkpoint/{args.dataset}/{detection_net_name}/{classifier_name}/{args.attack}')

        detection_net.eval()
        true = []
        pred = []
        valid_loss = 0
        for i, (x_original, x_adv) in enumerate(zip(valid_loader, valid_adv_loader)):
            # get the inputs; data is a list of [inputs, labels]
            x, x_adv = x_original[0].to(device), x_adv[0].to(device)
            labels = torch.cat((torch.zeros(x.shape[0], dtype=torch.long),
                                torch.ones(x_adv.shape[0], dtype=torch.long))).to(device)
            x_input = torch.cat((x, x_adv), dim=0).to(device)
            lhe_image = local_histogram_equalization(x_input, 8).to(device)

            x_input_4low = F.interpolate(
                F.interpolate(lhe_image, scale_factor=1 / 2, mode='bicubic', align_corners=False,
                              recompute_scale_factor=True), scale_factor=2, mode='bicubic', align_corners=False,
                recompute_scale_factor=True).to(device)

            x_input_4high = lhe_image - x_input_4low
            x_input_4high.to(device)
            x_input_4low = torch.autograd.Variable(x_input_4low, requires_grad=True)
            y_logit = classifier(x_input_4low)
            pred_class = y_logit.argmax(dim=1)
            loss = criterion(y_logit, pred_class)
            gradient = torch.autograd.grad(loss, x_input_4low)[0]
            gradient = torch.abs(gradient).detach().cuda()
            with torch.no_grad():
                mini_out = detection_net(x_input_4high, gradient)
                # outputs = nn.Softmax(dim=1)(outputs)
                loss = criterion_loss(mini_out, labels)
                pred.append(mini_out.argmax(dim=1))
                true.append(labels)
                valid_loss += len(x_input) * loss
        true = torch.cat(true, dim=0)
        pred = torch.cat(pred, dim=0)
        correct_predictions = pred.eq(true).sum()
        accuracy = correct_predictions / (2 * len(valid_loader.dataset)) * 100
        v_loss_mean = valid_loss.cpu().numpy() / (2 * len(valid_loader.dataset))
        losses.append(v_loss_mean)
        if best_accuracy <= accuracy:
            best_accuracy = accuracy
            torch.save(detection_net.state_dict(),
                       f"{base_dir}/checkpoint/{args.dataset}/{detection_net_name}/{classifier_name}/{args.attack}/eps2_{eps}_train.pth")
        print('validation acc : {:.2f}% \t validation loss : {:.4f}'.format(
            accuracy.cpu().numpy(), v_loss_mean))

    print('Finished Training')

    for s_type, loader in zip(['test_normal', 'test_adversarial'], [test_loader, test_adv_loader]):
        if s_type == 'test_normal':
            _, acc, test_pred, test_true = evaluate(classifier, loader, criterion, device, return_pred=True)
            print("Model accuracy on the %s test set: %0.2f%%" % (s_type, acc))
        else:
            _, acc = evaluate(classifier, loader, criterion, device)
            print("Model accuracy on the %s(eps is: %d/255) %s test set: %0.2f%%" % (args.attack, eps, s_type, acc))

    test_inds_correct = torch.where((test_true == test_pred))[0]
    test_loader = get_subsample_loader(args, test_loader, test_inds_correct)
    test_adv_loader = get_subsample_loader(args, test_adv_loader, test_inds_correct)

    detection_net.eval()
    detection_net.load_state_dict(torch.load(
        f"{base_dir}/checkpoint/{args.dataset}/{detection_net_name}/{classifier_name}/{args.attack}/eps2_{eps}_train.pth"))
    true = []
    pred = []
    probs = []
    for i, (x_original, x_adv) in enumerate(zip(test_loader, test_adv_loader)):
        # get the inputs; data is a list of [inputs, labels]
        x, x_adv = x_original[0].to(device), x_adv[0].to(device)
        labels = torch.cat((torch.zeros(x.shape[0], dtype=torch.long),
                            torch.ones(x_adv.shape[0], dtype=torch.long))).to(device)
        x_input = torch.cat((x, x_adv), dim=0).to(device)
        lhe_image = local_histogram_equalization(x_input, 8).to(device)

        x_input_4low = F.interpolate(
            F.interpolate(lhe_image, scale_factor=1 / 2, mode='bicubic', align_corners=False,
                          recompute_scale_factor=True), scale_factor=2, mode='bicubic', align_corners=False,
            recompute_scale_factor=True).to(device)  # 4倍high-pass

        x_input_4high = lhe_image - x_input_4low
        x_input_4high.to(device)
        x_input_4low = torch.autograd.Variable(x_input_4low, requires_grad=True)
        y_logit = classifier(x_input_4low)
        pred_class = y_logit.argmax(dim=1)
        loss = criterion(y_logit, pred_class)
        gradient = torch.autograd.grad(loss, x_input_4low)[0]
        gradient = torch.abs(gradient).detach().cuda()

        with torch.no_grad():
            mini_out = detection_net(x_input_4high, gradient)
            pred.append(mini_out.argmax(dim=1))
            true.append(labels)
            probs.append(mini_out[:, 1])
    true = torch.cat(true, dim=0)
    pred = torch.cat(pred, dim=0)
    probs = torch.cat(probs, dim=0).cpu().numpy()

    fpr, tpr, auc_score = compute_roc(probs, true.cpu().numpy(), plot=True)
    print(f'Detector is lhl, dataset is: {args.dataset}, classifier is {classifier_name}, '
          f'attack method is: {args.attack}_esp_{eps}')
    print("Detector accuracy on the %s test set: %0.4f" % (args.dataset, accuracy_score(true.cpu().numpy(), pred.cpu().numpy())))
    print("Detector Precision on the %s test set: %0.4f" % (args.dataset, precision_score(true.cpu().numpy(), pred.cpu().numpy())))
    print("Detector recall is: %f" % recall_score(true.cpu().numpy(), pred.cpu().numpy()))
    print('Detector ROC-AUC score: %0.4f' % auc_score)

    print('-------------------------------------------------------------------------------------------')


class Args():
    dataset = 'cifar'
    classifier = 'vgg16'
    attack = 'PGD'
    batch_size = 64
    epochs = 30
    epsilon = 12


if __name__ == "__main__":
    # parser.add_argument(
    #     '-d', '--dataset',
    #     help="Dataset to use; either 'mnist', 'cifar' or 'svhn'",
    #     default='mnist', required=True, type=str
    # )
    # parser.add_argument(
    #     '-a', '--attack',
    #     help="Attack to use; either 'fgsm', 'bim-a', 'bim-b', 'jsma' 'cw' "
    #          "or 'all'",
    #     required=True, type=str
    # )
    # parser.add_argument(
    #     '-e', '--epochs',
    #     help="The number of epochs to train for.",
    #     required=False, type=int
    # )
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     '-m', '--model',
    #     help="classifier model; torch_nets in net",
    #     required=True, type=str
    # )
    # parser.add_argument(
    #     '-b', '--batch_size',
    #     help="The batch size to use for training.",
    #     required=False, type=int
    # )
    # parser.add_argument(
    #     '-r', '--epsilon',
    #     help="The epsilon of adversarial attack.",
    #     required=False, type=int
    # )
    # parser.set_defaults(batch_size=256)
    # args = parser.parse_args()

    args = Args()
    main(args)