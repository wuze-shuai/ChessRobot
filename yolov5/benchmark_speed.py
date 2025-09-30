#!/usr/bin/env python3
"""
benchmark_speed.py
对比 YOLOv5  best.pt  vs  best.onnx  的推理延迟
> python benchmark_speed.py  --weights best.pt  --onnx best.onnx  --img test.jpg
"""
import argparse
import time
import cv2
import numpy as np
import torch
import onnxruntime as ort
from tqdm import tqdm

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    # 测速专用：直接 resize 到 640×640
    return cv2.resize(im, new_shape, interpolation=cv2.INTER_LINEAR), 1.0, (0, 0)

# ---------- PyTorch ----------
def load_pt(weights, device):
    model = torch.load(weights, map_location=device)['model'].float().fuse().eval()
    model = model.to(device)
    return model

def infer_pt(model, img, device):
    img = torch.from_numpy(img).to(device)
    img = img.float() / 255.0  # 0-1
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    with torch.no_grad():
        out = model(img)
    return out

# ---------- ONNX ----------
def load_onnx(path):
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if ort.get_device() == 'GPU' else ['CPUExecutionProvider']
    sess = ort.InferenceSession(path, providers=providers)
    return sess

def infer_onnx(sess, img):
    input_name = sess.get_inputs()[0].name
    out = sess.run(None, {input_name: img})
    return out

# ---------- 主流程 ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default='best.pt', help='best.pt')
    parser.add_argument('--onnx',default='best_int8.onnx',help='best.onnx')
    parser.add_argument('--img', default='test.jpg',help='test image')
    parser.add_argument('--num', type=int, default=100, help='benchmark loops')
    parser.add_argument('--device', default='cpu' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    im0 = cv2.imread(args.img)
    im, _, _ = letterbox(im0)
    im = im.transpose((2, 0, 1))[::-1]  # HWC→CHW, BGR→RGB
    im = np.ascontiguousarray(np.expand_dims(im, 0))

    # -------------- PyTorch --------------
    print('Loading .pt...')
    model_pt = load_pt(args.weights, args.device)
    # 预热
    for _ in range(10):
        _ = infer_pt(model_pt, im, args.device)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in tqdm(range(args.num), desc='PyTorch'):
        _ = infer_pt(model_pt, im, args.device)
    torch.cuda.synchronize()
    pt_time = (time.perf_counter() - t0) / args.num * 1000
    print(f'[PyTorch]  avg latency: {pt_time:.2f} ms   FPS: {1000/pt_time:.1f}')

    # -------------- ONNX --------------
    print('Loading .onnx...')
    sess = load_onnx(args.onnx)
    # 预热
    for _ in range(10):
        _ = infer_onnx(sess, im.astype(np.float32))
    t0 = time.perf_counter()
    for _ in tqdm(range(args.num), desc='ONNXRuntime'):
        _ = infer_onnx(sess, im.astype(np.float32))
    onnx_time = (time.perf_counter() - t0) / args.num * 1000
    print(f'[ONNX]     avg latency: {onnx_time:.2f} ms   FPS: {1000/onnx_time:.1f}')

    # -------------- 对比 --------------
    print(f'\nSpeed-up: {pt_time/onnx_time:.2f}x  (ONNX vs PyTorch)')

if __name__ == '__main__':
    main()