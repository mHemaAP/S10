import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt

##### Get Device Details #####
def get_device() -> tuple:
    """Get Device type

    Returns:
        tuple: Device type
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    return (use_cuda, device)

# def move_loss_to_cpu(loss):
#   moved_loss2cpu = [t.cpu().item() for t in loss]
#   return moved_loss2cpu

#####  Get the count of correct predictions
def GetCorrectPredCount(pPrediction, pLabels):
    return pPrediction.argmax(dim=1).eq(pLabels).sum().item()


test_incorrect_pred = {"images": [], "ground_truths": [], "predicted_vals": []}


#####  Get the incorrect predictions
def GetInCorrectPreds(pPrediction, pLabels):
    pPrediction = pPrediction.argmax(dim=1)
    indices = pPrediction.ne(pLabels).nonzero().reshape(-1).tolist()
    return indices, pPrediction[indices].tolist(), pLabels[indices].tolist()


def get_incorrect_test_predictions(model, device, test_loader):
    model.eval()

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)

            ind, pred, truth = GetInCorrectPreds(output, target)
            test_incorrect_pred["images"] += data[ind]
            test_incorrect_pred["ground_truths"] += truth
            test_incorrect_pred["predicted_vals"] += pred

    return test_incorrect_pred

#####  Display the shape and decription of the train data
def display_train_data(train_data):

  print('[Train]')
  print(' - Numpy Shape:', train_data.cpu().numpy().shape)
  print(' - Tensor Shape:', train_data.size())
  print(' - min:', torch.min(train_data))
  print(' - max:', torch.max(train_data))
  print(' - mean:', torch.mean(train_data))
  print(' - std:', torch.std(train_data))
  print(' - var:', torch.var(train_data))


"""
args: 
points should be of type list of tuples or lists
Ex : [{[x: xpoints, y: ypoints, label: title, xlabel: "" , ylabel: " "}),([....points],label)]
"""
def plot_learning_rate_trend(curves,title,Figsize = (7,7)):
    fig = plt.figure(figsize=Figsize)
    ax = plt.subplot()
    for curve in curves:
        if("x" not in curve):
            ax.plot(curve["y"], label=curve.get("label", "label"))   
        else:
            ax.plot(curve["x"],curve["y"], label=curve.get("label","label"))
        plt.xlabel(curve.get("xlabel","x-axis"))
        plt.ylabel(curve.get("ylabel","y-axis"))
        plt.title(title)
    ax.legend()
    plt.show()