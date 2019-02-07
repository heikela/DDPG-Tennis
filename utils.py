def soft_update(local_model, target_model, tau):
    """Soft update model parameters.
    target parameters = tau * local parameters + (1 - tau) * target parameters
    """
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(tau*local_param.data + (1.0 - tau) * target_param.data)


def mutate_param(param, grow_factor=2.):
    """A utility for randomly adjusting hyperparameters in the explore step of the PBT process"""
    choice = np.random.randint(-1, 1)
    if choice == -1:
        return param / grow_factor
    if choice == 0:
        return param
    return param * grow_factor
