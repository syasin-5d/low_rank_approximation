import numpy as np
from PIL import Image
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file_name', type=str, required=True)
    parser.add_argument('-s', '--size', type=int)
    parser.add_argument('-k', type=int, required=True)
    args = parser.parse_args()
    return args


def load_image(path):
    img = np.array(Image.open(path).convert('L'))
    return img


def export_image(M_k, k):
    img2 = Image.fromarray(np.uint8(M_k))
    out_name = f'out_{k}.jpg'
    img2.save(out_name)
    print(f'{out_name} was generated.')


def matrix2singulars(M):
    import numpy.linalg as LA
    u, s, v = LA.svd(M)
    return u, s, v


def low_rank_approximation(k, u, s, v):
    ur = u[:, :k - 1]
    sr = np.diag(s[:k - 1])
    vr = v[:k - 1, :]
    Mk = np.dot(np.dot(ur, sr), vr)
    return Mk


def main():
    args = parse_args()
    image_path = args.file_name
    img = load_image(image_path)
    u, s, v = matrix2singulars(img)
    k = args.k
    Mk = low_rank_approximation(k, u, s, v)
    export_image(Mk, k)


if __name__ == '__main__':
    main()
