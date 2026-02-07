"""
간단한 3D 에피소드 시각화 도구

사용법:
  python visualize/visualize_episode_3d.py --input data/episodes/<file>.npz

옵션:
  --key : npz 내부에서 3D 키포인트 배열을 직접 지정 (기본 자동탐색)
  --start --end : 프레임 범위 (인덱스)
  --interval : 애니메이션 프레임 간격(ms)
  --save : 출력 파일(mp4)로 저장 (ffmpeg 필요)

동작:
 - .npz 파일을 로드한 뒤 내부에서 3차원 좌표(T, J, 3) 또는 2차원 좌표(T, J*3) 형태를 자동으로 탐색합니다.
 - 찾은 키포인트를 3D 산점/라인으로 애니메이션합니다. 관절 연결 정보가 주어지지 않으면 인덱스 순서로 선을 연결합니다.
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib import animation
import sys


def load_npz_candidate(path: Path, key: str = None):
    data = np.load(path, allow_pickle=True)
    candidates = {}
    for k in data.files:
        arr = data[k]
        if isinstance(arr, np.ndarray):
            # candidate if 3D coords or flattened coords
            if arr.ndim == 3 and arr.shape[2] in (2, 3):
                candidates[k] = arr
            elif arr.ndim == 2:
                # T x (J*3) or (T*J) x 3 heuristics
                if arr.shape[1] % 3 == 0:
                    T = arr.shape[0]
                    J = arr.shape[1] // 3
                    candidates[k] = arr.reshape((T, J, 3))
                elif arr.shape[1] % 2 == 0:
                    # maybe 2D keypoints
                    T = arr.shape[0]
                    J = arr.shape[1] // 2
                    candidates[k] = np.concatenate([arr.reshape((T, J, 2)), np.zeros((T, J, 1))], axis=2)
    # also check top-level array if file contains single array (e.g., np.load returns array)
    if key:
        if key in candidates:
            return candidates[key], key
        # try to load key directly from file object if exists
        if key in data.files:
            arr = data[key]
            return _normalize_array(arr), key
        raise ValueError(f"Key '{key}' not found in {path}. Available: {list(data.files)}")

    if not candidates:
        # try common names
        for common in ("poses", "keypoints", "joints", "pose3d", "positions", "xyz"):
            if common in data.files:
                arr = data[common]
                return _normalize_array(arr), common
        raise ValueError(f"No 3D candidate arrays found in {path}. Keys: {list(data.files)}")

    # pick largest candidate (most joints * frames)
    best_key = max(candidates.keys(), key=lambda k: candidates[k].size)
    return candidates[best_key], best_key


def _normalize_array(arr: np.ndarray):
    if arr.ndim == 3 and arr.shape[2] in (2, 3):
        if arr.shape[2] == 2:
            arr = np.concatenate([arr, np.zeros((arr.shape[0], arr.shape[1], 1))], axis=2)
        return arr
    if arr.ndim == 2 and arr.shape[1] % 3 == 0:
        T = arr.shape[0]
        J = arr.shape[1] // 3
        return arr.reshape((T, J, 3))
    raise ValueError("Unsupported array shape for normalization: %s" % (arr.shape,))


def auto_limits(points):
    # points: (T, J, 3)
    mins = np.nanmin(points.reshape(-1, 3), axis=0)
    maxs = np.nanmax(points.reshape(-1, 3), axis=0)
    # pad
    ranges = maxs - mins
    pad = ranges.max() * 0.1 if ranges.max() > 0 else 1.0
    mins -= pad
    maxs += pad
    return mins, maxs


def plot_animation(points, connections=None, interval=50, start=0, end=None, save_path=None):
    # points: (T, J, 3)
    T, J, _ = points.shape
    if end is None or end > T:
        end = T
    frames = end - start

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    mins, maxs = auto_limits(points[start:end])
    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[2], maxs[2])

    scat = ax.scatter([], [], [], c='r', s=30)
    lines = []
    if connections is None:
        # default: connect in index order as a single chain
        connections = [(i, i + 1) for i in range(J - 1)]
    for _ in connections:
        line, = ax.plot([], [], [], c='k', lw=2)
        lines.append(line)

    title = ax.set_title('')

    def init():
        scat._offsets3d = ([], [], [])
        for line in lines:
            line.set_data([], [])
            line.set_3d_properties([])
        title.set_text('')
        return [scat] + lines + [title]

    def update(i):
        idx = start + i
        pts = points[idx]
        xs = pts[:, 0]
        ys = pts[:, 1]
        zs = pts[:, 2]
        scat._offsets3d = (xs, ys, zs)
        for li, (a, b) in enumerate(connections):
            xa = [pts[a, 0], pts[b, 0]]
            ya = [pts[a, 1], pts[b, 1]]
            za = [pts[a, 2], pts[b, 2]]
            lines[li].set_data(xa, ya)
            lines[li].set_3d_properties(za)
        title.set_text(f'Frame {idx}/{end-1}')
        return [scat] + lines + [title]

    ani = animation.FuncAnimation(fig, update, frames=frames, init_func=init, blit=False, interval=interval)

    if save_path:
        try:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=1000/interval, metadata=dict(artist='p-ade'), bitrate=2000)
            ani.save(save_path, writer=writer)
            print(f"Saved animation to {save_path}")
        except Exception as e:
            print(f"Failed to save animation: {e}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize 3D episode .npz')
    parser.add_argument('--input', '-i', required=True, help='Path to .npz episode file')
    parser.add_argument('--key', help='Specific array key inside npz to use')
    parser.add_argument('--start', type=int, default=0, help='Start frame index')
    parser.add_argument('--end', type=int, default=None, help='End frame index (exclusive)')
    parser.add_argument('--interval', type=int, default=50, help='Animation interval (ms)')
    parser.add_argument('--save', help='Save animation as mp4 (requires ffmpeg)')
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    try:
        arr, used_key = load_npz_candidate(path, key=args.key)
    except Exception as e:
        print(f"Failed to find 3D data in {path}: {e}")
        sys.exit(1)

    print(f"Using array key: {used_key}, shape: {arr.shape}")

    # arr is (T, J, 3)
    plot_animation(arr, interval=args.interval, start=args.start, end=args.end, save_path=args.save)


if __name__ == '__main__':
    main()
