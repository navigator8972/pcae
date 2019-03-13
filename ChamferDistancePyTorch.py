import torch

def chamfer_distance_with_batch(p1, p2, verbose=False):

    '''
    Calculate Chamfer Distance between two point sets
    the arrangement of axes is different from the original implementation as we take coordiantes as channels
    :param p1: size[B, D, N]
    :param p2: size[B, D, M]
    :param debug: whether need to output debug info
    :return: sum of all batches of Chamfer Distance of two point sets
    '''

    assert p1.size(0) == p2.size(0) and p1.size(1) == p2.size(1)

    if verbose:
        print('num of pointsets: ', p1[0])

    p1 = p1.unsqueeze(1)
    p2 = p2.unsqueeze(1)
    if verbose:
        print('p1 size is {}'.format(p1.size()))
        print('p2 size is {}'.format(p2.size()))

    p1 = p1.repeat(1, p2.size(3), 1, 1)
    if verbose:
        print('p1 size is {}'.format(p1.size()))

    p1 = p1.transpose(1, 3)
    if verbose:
        print('p1 size is {}'.format(p1.size()))

    p2 = p2.repeat(1, p1.size(1), 1, 1)
    if verbose:
        print('p2 size is {}'.format(p2.size()))

    dist = torch.add(p1, torch.neg(p2))
    if verbose:
        print('dist size is {}'.format(dist.size()))
        print(dist[0])

    dist = torch.norm(dist, 2, dim=2)
    if verbose:
        print('dist size is {}'.format(dist.size()))
        print(dist)

    dist_p1_p2 = torch.min(dist, dim=2)[0].mean(dim=1)
    dist_p2_p1 = torch.min(dist, dim=1)[0].mean(dim=1)
    
    return dist_p1_p2, dist_p2_p1

if __name__ == '__main__':
    import numpy as np
    from sklearn.neighbors import KDTree

    batch_size = 8
    num_point = 20
    num_features = 4
    np.random.seed(1)
    array1 = np.random.randint(0, high=4, size=(batch_size, num_point, num_features)).astype(np.float32)
    array2 = np.random.randint(0, high=4, size=(batch_size, num_point, num_features)).astype(np.float32)

    array1_for_torch = np.swapaxes(array1, 1, 2)
    array2_for_torch = np.swapaxes(array2, 1, 2)

    def chamfer_distance_sklearn(array1,array2):
        batch_size, num_point = array1.shape[:2]
        dist = 0
        for i in range(batch_size):
            tree1 = KDTree(array1[i], leaf_size=num_point+1)
            tree2 = KDTree(array2[i], leaf_size=num_point+1)
            distances1, _ = tree1.query(array2[i])
            distances2, _ = tree2.query(array1[i])
            av_dist1 = np.mean(distances1)
            av_dist2 = np.mean(distances2)
            dist = dist + (av_dist1+av_dist2)/batch_size
        return dist
    

    sklearn_dist = chamfer_distance_sklearn(array1, array2)
    print('sklearn: ', sklearn_dist)

    def array2samples_distance(array1, array2):
        """
        arguments: 
            array1: the array, size: (num_point, num_feature)
            array2: the samples, size: (num_point, num_feature)
        returns:
            distances: each entry is the distance from a sample to array1 
        """
        num_point, num_features = array1.shape
        expanded_array1 = np.tile(array1, (num_point, 1))
        expanded_array2 = np.reshape(
                np.tile(np.expand_dims(array2, 1), 
                        (1, num_point, 1)),
                (-1, num_features))
        distances = np.linalg.norm(expanded_array1-expanded_array2, axis=1)
        distances = np.reshape(distances, (num_point, num_point))
        distances = np.min(distances, axis=1)
        distances = np.mean(distances)
        return distances

    def chamfer_distance_numpy(array1, array2):
        batch_size, _, _ = array1.shape
        dist = 0
        for i in range(batch_size):
            av_dist1 = array2samples_distance(array1[i], array2[i])
            av_dist2 = array2samples_distance(array2[i], array1[i])
            dist = dist + (av_dist1+av_dist2)/batch_size
        return dist

    numpy_dist = chamfer_distance_numpy(array1, array2)
    print('numpy: ', numpy_dist)

    dist_1_to_2, dist_2_to_1 = chamfer_distance_with_batch(torch.from_numpy(array1_for_torch), torch.from_numpy(array2_for_torch))
    print('pytorch: ', (dist_1_to_2+dist_2_to_1).mean().item())