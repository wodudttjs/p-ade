import numpy as np
import matplotlib.pyplot as plt

# 파일 경로 예시
file_path = 'data/episodes/2dqh3OLxt6E_episode.npz'

# npz 파일 로드
data = np.load(file_path)

# 내부 키 확인
print('keys:', data.files)

# 예시: 'poses' 키가 있다면 3D 시각화
if 'poses' in data:
    poses = data['poses']  # shape: (N, 33, 3) 등
    # 첫 프레임의 모든 랜드마크 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(poses[0,:,0], poses[0,:,1], poses[0,:,2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
else:
    print('poses 키가 없습니다.')