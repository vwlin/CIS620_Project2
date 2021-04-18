"""
    [1] https://github.com/learnables/learn2learn/blob/master/examples/vision/maml_omniglot.py
"""

"""
Demonstrates how to:
    * use the MAML wrapper for fast-adaptation,
    * use the benchmark interface to load CURE TSR, and
    * sample tasks and split them in adaptation and evaluation sets.
"""

import random
import numpy as np
import torch
import torchvision
import learn2learn as l2l
from torch import nn, optim

# my custom imports
from CURE_TSR_tasksets import get_cure_tsr_tasksets
from cure_tasksets_bp import get_cure_tsr_inter_tasksets
from models import LeNet5

def student_func(teacher):
    return teacher

def self_train_once(student, teacher, unsup_x, confidence_q=0.1, epochs=20):
    # Do one bootstrapping step on unsup_x, where pred_model is used to make predictions,
    # and we use these predictions to update model.
    logits = teacher.predict(np.concatenate([unsup_x]))
    confidence = np.amax(logits, axis=1) - np.amin(logits, axis=1)
    alpha = np.quantile(confidence, confidence_q)
    indices = np.argwhere(confidence >= alpha)[:, 0]
    preds = np.argmax(logits, axis=1)
    student.fit(unsup_x[indices], preds[indices], epochs=epochs, verbose=False)

def soft_self_train_once(student, teacher, unsup_x, epochs=20):
    probs = teacher.predict(np.concatenate([unsup_x]))
    student.fit(unsup_x, probs, epochs=epochs, verbose=False)

def gradual_self_train(student_func, teacher, unsup_x, debug_y, interval, confidence_q=0.1,
                       epochs=20, soft=False):
    upper_idx = int(unsup_x.shape[0] / interval)
    accuracies = []
    for i in range(upper_idx):
        student = student_func(teacher)
        cur_xs = unsup_x[interval*i:interval*(i+1)]
        cur_ys = debug_y[interval*i:interval*(i+1)]
        if soft:
            soft_self_train_once(student, teacher, cur_xs, epochs)
        else:
            self_train_once(student, teacher, cur_xs, confidence_q, epochs)
        _, accuracy = student.evaluate(cur_xs, cur_ys)
        accuracies.append(accuracy)
        teacher = student
    return accuracies, student


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)

def training_data(batch, shots, ways, device):
    data, labels = batch
    data, labels = data.to(device), labels.to(device)

    # Separate data into adaptation/evalutation (a.k.a., support / query) sets
    adaptation_indices = np.zeros(data.size(0), dtype=bool)
    adaptation_indices[np.arange(shots*ways) * 2] = True
    evaluation_indices = torch.from_numpy(~adaptation_indices)
    adaptation_indices = torch.from_numpy(adaptation_indices)
    adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
    evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]

    return adaptation_data,evaluation_data,adaptation_labels,evaluation_labels

def fast_adapt(batch, learner, loss, adaptation_steps, shots, ways, device):
    data, labels = batch
    data, labels = data.to(device), labels.to(device)

    # Separate data into adaptation/evalutation (a.k.a., support / query) sets
    adaptation_indices = np.zeros(data.size(0), dtype=bool)
    adaptation_indices[np.arange(shots*ways) * 2] = True
    evaluation_indices = torch.from_numpy(~adaptation_indices)
    adaptation_indices = torch.from_numpy(adaptation_indices)
    adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
    evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]

    # Adapt the model
    for step in range(adaptation_steps):
        train_error = loss(learner(adaptation_data), adaptation_labels)
        learner.adapt(train_error)

    # Evaluate the adapted model
    predictions = learner(evaluation_data)
    valid_error = loss(predictions, evaluation_labels)
    valid_accuracy = accuracy(predictions, evaluation_labels)
    return valid_error, valid_accuracy

def fast_adapt_generate_label(batch, learner, adaptation_data, evaluation_data, adaptation_labels,evaluation_labels, shots, ways, device):
    # Evaluate the adapted model
    predictions = learner(adaptation_data)
    adaptation_pseudo_labels = predictions.argmax(dim=1).view(adaptation_labels.shape)
    #print("adaptation labels",adaptation_pseudo_labels)
    # Evaluate the adapted model
    predictions = learner(evaluation_data)
    evaluation_pseudo_labels = predictions.argmax(dim=1).view(evaluation_labels.shape)
    #print("evaulation labels",evaluation_pseudo_labels)
    return adaptation_pseudo_labels, evaluation_pseudo_labels

def fast_adapt_with_pseudo_label(batch, learner, loss, adaptation_steps, shots, ways, device,adaptation_data, evaluation_data,adaptation_labels, evaluation_labels):
    adaptation_data, adaptation_labels = adaptation_data.to(device), adaptation_labels.to(device)
    evaluation_data, evaluation_labels = evaluation_data.to(device), evaluation_labels.to(device)

    # Adapt the model
    for step in range(adaptation_steps):
        train_error = loss(learner(adaptation_data), adaptation_labels)
        learner.adapt(train_error)

    # Evaluate the adapted model
    predictions = learner(evaluation_data)
    valid_error = loss(predictions, evaluation_labels)
    valid_accuracy = accuracy(predictions, evaluation_labels)
    return valid_error, valid_accuracy

def main(
        ways=5,
        shots=1,
        meta_lr=0.003,
        fast_lr=0.5,
        meta_batch_size=32,
        adaptation_steps=1,
        num_iterations=151, # originally, 60000
        cuda=True,
        seed=42,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cpu')
    if cuda:
        torch.cuda.manual_seed(seed)
        device = torch.device('cuda')

    # Load train/validation/test tasksets using the benchmark interface
    tasksets = get_cure_tsr_tasksets(train_ways=ways,
                                                  train_samples=2*shots,
                                                  test_ways=ways,
                                                  test_samples=2*shots,
                                                  num_tasks=10000, # originally, 20000
    )

    # Create model
    #model = LeNet5(num_labels = 14)
    model = torchvision.models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 14)  

    model.to(device)
    maml = l2l.algorithms.MAML(model, lr=fast_lr, first_order=False)
    opt = optim.Adam(maml.parameters(), meta_lr)
    loss = nn.CrossEntropyLoss(reduction='mean')
    check_int = 1
    levels = 5

    for iteration in range(num_iterations):
        opt.zero_grad()
        meta_train_error = 0.0
        meta_train_accuracy = 0.0
        meta_valid_error = 0.0
        meta_valid_accuracy = 0.0
        for task in range(meta_batch_size):
            # Compute meta-training loss
            learner = maml.clone()
            batch = tasksets.train.sample()
            if check_int:
                print("==> Training: batch shape X={}, Y={} and dataset length {}".format(batch[0].shape, batch[1].shape, len(tasksets.train)))
            evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                               learner,
                                                               loss,
                                                               adaptation_steps,
                                                               shots,
                                                               ways,
                                                               device)
            evaluation_error.backward()
            meta_train_error += evaluation_error.item()
            meta_train_accuracy += evaluation_accuracy.item()

            # Compute meta-validation loss
            learner = maml.clone()
            batch = tasksets.validation.sample()
            if check_int:
                print("==> Validation: batch shape X={}, Y={} and dataset length {}".format(batch[0].shape, batch[1].shape, len(tasksets.validation)))
                check_int = 0
            evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                               learner,
                                                               loss,
                                                               adaptation_steps,
                                                               shots,
                                                               ways,
                                                               device)
            meta_valid_error += evaluation_error.item()
            meta_valid_accuracy += evaluation_accuracy.item()

        # Print some metrics
        print('\n')
        print('Iteration', iteration)
        print('Meta Train Error', meta_train_error / meta_batch_size)
        print('Meta Train Accuracy', meta_train_accuracy / meta_batch_size)
        print('Meta Valid Error', meta_valid_error / meta_batch_size)
        print('Meta Valid Accuracy', meta_valid_accuracy / meta_batch_size)

        # Average the accumulated gradients and optimize
        for p in maml.parameters():
            p.grad.data.mul_(1.0 / meta_batch_size)
        opt.step()


    for level in range(1,levels):
        inter_dir = './CURE_TSR_Yahan_Shortcut/Real_Train/Snow-' + str(level)+'/'
        print("level :", level, inter_dir)
        intermediate_tasksets = get_cure_tsr_inter_tasksets( inter_dir = inter_dir,
                                                  train_ways=ways,
                                                  train_samples=2*shots,
                                                  test_ways=ways,
                                                  test_samples=2*shots,
                                                  num_tasks=10000, # originally, 20000
    )
        for iteration in range(num_iterations):
            opt.zero_grad()
            meta_train_error = 0.0
            meta_train_accuracy = 0.0
            meta_valid_error = 0.0
            meta_valid_accuracy = 0.0
            for task in range(meta_batch_size):
                # Compute meta-training loss
                learner = maml.clone()
                batch = intermediate_tasksets.train.sample()
                if check_int:
                    print("==> Intermediate Training: batch shape X={}, Y={} and dataset length {}".format(batch[0].shape, batch[1].shape, len(tasksets.train)))
                adaptation_data,evaluation_data,adaptation_labels,evaluation_labels = training_data(batch, shots, ways, device)
                adaptation_pseudo_labels, evaluation_pseudo_labels = fast_adapt_generate_label(batch, learner, adaptation_data, evaluation_data, adaptation_labels,evaluation_labels,shots, ways, device)
                evaluation_error, evaluation_accuracy = fast_adapt_with_pseudo_label(batch,
                                                                learner,
                                                                loss,
                                                                adaptation_steps,
                                                                shots,
                                                                ways,
                                                                device,
                                                                adaptation_data, evaluation_data,adaptation_pseudo_labels, evaluation_pseudo_labels)
                evaluation_error.backward()
                meta_train_error += evaluation_error.item()
                meta_train_accuracy += evaluation_accuracy.item()

                # Compute meta-validation loss
                learner = maml.clone()
                batch = intermediate_tasksets.validation.sample()
                if check_int:
                    print("==> Intermediate Validation: batch shape X={}, Y={} and dataset length {}".format(batch[0].shape, batch[1].shape, len(tasksets.validation)))
                    check_int = 0
                evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                                learner,
                                                                loss,
                                                                adaptation_steps,
                                                                shots,
                                                                ways,
                                                                device)
                meta_valid_error += evaluation_error.item()
                meta_valid_accuracy += evaluation_accuracy.item()

            # Print some metrics
            print('\n')
            print('BIG ISSUE --> valida error is Nan? {}'.format(np.isnan(meta_valid_error)))
            
            print('Intermediate Iteration', iteration)
            print('Intermediate Meta Train Error', meta_train_error / meta_batch_size)
            print('Intermediate Meta Train Accuracy', meta_train_accuracy / meta_batch_size)
            print('Intermediate Meta Valid Error', meta_valid_error / meta_batch_size)
            print('Intermediate Meta Valid Accuracy', meta_valid_accuracy / meta_batch_size)

            # Average the accumulated gradients and optimize
            for p in maml.parameters():
                p.grad.data.mul_(1.0 / meta_batch_size)
            opt.step()

    meta_test_error = 0.0
    meta_test_accuracy = 0.0
    for task in range(meta_batch_size):
        # Compute meta-testing loss
        learner = maml.clone()
        batch = tasksets.test.sample()
        evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                           learner,
                                                           loss,
                                                           adaptation_steps,
                                                           shots,
                                                           ways,
                                                           device)
        meta_test_error += evaluation_error.item()
        meta_test_accuracy += evaluation_accuracy.item()
    print('Meta Test Error', meta_test_error / meta_batch_size)
    print('Meta Test Accuracy', meta_test_accuracy / meta_batch_size)


if __name__ == '__main__':
    main()