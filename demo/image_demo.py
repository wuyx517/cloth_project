import asyncio
from argparse import ArgumentParser
import cv2
import numpy as np
from mmdet.apis import inference_detector, init_detector


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--out-file', default='demo.png', help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    args = parser.parse_args()
    return args


def main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    result = inference_detector(model, args.img)
    # print(result[0][0].shape)
    # print(result[1][0].shape)
    image = cv2.imread(args.img)
    print(result[0][1])
    print(result[1][1][0])
    # for i in range(len(result[1])):
    zeos_image = np.zeros(image.shape[:2])
    #     if len(result[1][i]) == 0:
    #         continue
    #     print(len(result[1][i]))
    zeos_image[result[1][1][0]] = 255
    cv2.imwrite(f'ts{1}.jpg', zeos_image)
    # show the results
    model.show_result(
        args.img,
        result[:2],
        score_thr=args.score_thr,
        out_file=args.out_file)


if __name__ == '__main__':
    args = parse_args()
    main(args)
