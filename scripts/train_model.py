from util import get_dataloader, get_imagenet_classifier, get_cifar_classifier, set_seed
import torch
import torch.nn as nn
from absl import app, flags
import os
import global_variable
base_dir = global_variable.base_dir
args = flags.FLAGS


def main(args):
    set_seed(0)
    assert args.dataset in ['mnist', 'cifar', 'cifar100', 'svhn', 'imagenet'], \
        "dataset parameter must be either 'mnist', 'cifar' or 'svhn'"
    print('Data set: %s' % args.dataset)
    
    train_loader, valid_loader, test_loader = get_dataloader(args)
    if args.dataset == 'imagenet':
        classifier, classifier_name = get_imagenet_classifier(args.classifier, pretrained=False)
    elif args.dataset == 'cifar100':
        classifier, classifier_name = get_cifar_classifier(args.classifier, num_classes=100, pretrained=False)
    else:
        classifier, classifier_name = get_cifar_classifier(args.classifier, num_classes=10, pretrained=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Used device : {device}         classifier is: {classifier_name}')

    if not os.path.isdir(f'{base_dir}/checkpoint/{args.dataset}'):
        os.makedirs(f'{base_dir}/checkpoint/{args.dataset}')

    EPOCHS = args.epochs
    lr = 1e-4

    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    classifier.to(device)

    losses = []
    best_v_loss = 10000.0
    for epoch in range(EPOCHS):  # loop over the dataset multiple times
        running_loss = 0.0
        classifier.train()
        for i, data in enumerate(train_loader):
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = classifier(inputs)
            # plot_image(inputs, high)
            # outputs = nn.Softmax(dim=1)(outputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            if (i+1) % 20 == 0:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 20:.3f}')
                running_loss = 0.0

        classifier.eval()
        with torch.no_grad():
            true = []
            pred = []
            valid_loss = 0
            for i, data in enumerate(valid_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(device), data[1].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = classifier(inputs)
                # outputs = nn.Softmax(dim=1)(outputs)
                loss = criterion(outputs, labels)

                pred.append(outputs.argmax(dim=1))
                true.append(labels)
                valid_loss += len(inputs)*loss
            true = torch.cat(true, dim=0)
            pred = torch.cat(pred, dim=0)
            correct_predictions = pred.eq(true).sum()
            accuracy = correct_predictions / len(valid_loader.dataset) * 100
            v_loss_mean = valid_loss.cpu().numpy() / len(valid_loader.dataset)
            losses.append(v_loss_mean)

            if v_loss_mean < best_v_loss:
                best_v_loss = v_loss_mean
                torch.save(classifier.state_dict(),
                           f"{base_dir}/checkpoint/{args.dataset}/classifier_{classifier_name}.pth")
        print('validation acc : {:.2f}% \t validation loss : {:.4f}'.format(
            accuracy.cpu().numpy(), v_loss_mean))

    print('Finished Training')


class Args():
    dataset = 'cifar'
    batch_size = 64
    epochs = 50
    classifier = 'vgg19'


if __name__ == "__main__":

    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     '-d', '--dataset',
    #     help="Dataset to use; either 'mnist', 'cifar' or 'svhn'",
    #     required=True, type=str,
    # )
    # parser.add_argument(
    #     '-m', '--model',
    #     help="classifier model; model in net",
    #     required=True, type=str,
    # )
    # parser.add_argument(
    #     '-e', '--epochs',
    #     help="The number of epochs to train for.",
    #     required=False, type=int
    # )
    # parser.add_argument(
    #     '-b', '--batch_size',
    #     help="The batch size to use for training.",
    #     required=False, type=int
    # )
    #
    # parser.set_defaults(epochs=20)
    # parser.set_defaults(batch_size=128)
    # args = parser.parse_args()


    args = Args()

    main(args)