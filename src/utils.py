import torch
import torch.nn.functional as F


# FGSM attack code from pytorch tutorials.
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image


# Attack a network, make sure batch_size = 1 for this block
def attack_model(net, dataloader, epsilon, Pretrained=None):
    if not Pretrained is None:
        net.load_state_dict(torch.load(Pretrained))
    correct = 0
    adv_samples   = []
    sample_labels = []
    for _, data in enumerate(dataloader):
        image, target = data
        image, target = image.cuda(), target.cuda()

        # Set requires_grad attribute of tensor.
        image.requires_grad = True

        # Forward pass the data through the model
        output = net(image)
        init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

        # If the initial prediction is wrong, dont bother attacking, just move on
        if init_pred.item() != target.item():
            continue

        # Calculate the loss
        loss = F.nll_loss(output, target)

        # Zero all existing gradients
        net.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect datagrad
        image_grad = image.grad.data

        # Call FGSM Attack
        perturbed_image = fgsm_attack(image, epsilon, image_grad)

        # Save perturbed_images
        adv_samples.append(perturbed_image.detach().cpu())
        sample_labels.append(target)

        # Re-classify the perturbed image
        output = net(perturbed_image)

        # Check for success
        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        if final_pred.item() == target.item():
            correct += 1


    # Calculate final accuracy for this epsilon
    final_acc = correct/float(len(dataloader))
    print("Epsilon: {}\tTest Accuracy = {}".format(epsilon, final_acc))

    # Return the accuracy and an adversarial example
    return adv_samples, sample_labels
