import os
import torch


def save_checkpoint(net, clf, critic, predictor, epoch, acc, acc2, args, script_name):
    # Save checkpoint.
    print('Saving..')
    state = {
        'net': net.state_dict(),
        'clf': clf.state_dict(),
        'critic': critic.state_dict(),
        'predictor': predictor.state_dict(),
        'epoch': epoch,
        'acc': acc,
        'acc2': acc2,
        'args': vars(args),
        'script': script_name
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    destination = os.path.join('./checkpoint', f'{args.filename}.pth')
    torch.save(state, destination)
