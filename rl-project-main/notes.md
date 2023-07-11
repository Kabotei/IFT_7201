# Notes

## Actor-Critic

[Source](https://arxiv.org/pdf/1512.07679.pdf)

**Steps :**

1. Send state as input to Actor
    1. State is a numpy array of (*current_node, *end_node) NORMALIZED
2. Actor outputs an action, with corresponds to an aproximation of next_node (pseudo_next_node)
    1. pseudo_next_node is a numpy array of (x, y) NORMALIZED
3. Find next_node from available actions that is nearest to pseudo_next_node
    1. As always, everything happens in the normalized space
4. Send ... -> how to use Critic with only 1 KNN?
