# 프로젝트 전체 설명

## 목적

이 프로젝트는 책상 또는 바스켓 형태의 작업 공간 위에 여러 물체가 놓인 상황에서 특정 target object를 찾기 위한 2D-PDM 기반 데이터 생성/후처리/학습 파이프라인입니다. 시뮬레이터가 RGB, depth, semantic segmentation 데이터를 생성하고, 후처리 스크립트가 물체 간 유사도와 가림 정보를 결합해 target 위치 추정에 사용할 distribution map을 만듭니다.

## 핵심 아이디어

- **합성 데이터 생성**: Isaac Sim 또는 Genesis에서 여러 물체를 랜덤/유사도 기반으로 배치한 뒤 카메라 관측 데이터를 저장합니다.
- **Target occlusion 분석**: target object를 여러 위치, 회전, 높이에서 관측해 가림 가능성과 depth 패턴을 수집합니다.
- **Similarity map 생성**: target category와 scene object category 간 유사도 점수를 segmentation image에 투영합니다.
- **Distribution map 생성**: similarity map, occlusion map, depth 기반 정보를 결합해 target이 있을 가능성이 높은 2D 영역을 계산합니다.
- **Segmentation model 학습**: 생성된 input/output map을 이용해 FCN-ResNet50 또는 DeepLabV3-ResNet50 기반 segmentation 모델을 학습합니다.

## 구현 백엔드

### Isaac Sim 파이프라인

`src/isaac`은 NVIDIA Isaac Sim 기반 구현입니다. 병렬 환경 생성, USD asset spawning, multi-view camera capture, GPU 기반 distribution map 후처리, segmentation training script를 포함합니다.

주요 특징:

- `vectorized_scene.py`가 여러 환경을 병렬로 구성하여 scene 데이터를 빠르게 생성합니다.
- `object_spawner.py`가 category별 USD/USDC asset을 탐색하고 workspace 위에 배치합니다.
- `object_occlusion.py`와 `vectorized_object_occlusion.py`가 target object의 가림/깊이 데이터를 생성합니다.
- `similarity_map_generator.py`와 `distribution_map_GPU.py`가 segmentation/depth 결과를 map 형태로 후처리합니다.
- `train_260222.py`가 PyTorch segmentation 모델 학습을 담당합니다.

### Genesis 파이프라인

`src/genesis`는 Genesis simulator 기반 구현입니다. Isaac Sim과 유사한 목적을 갖지만, asset category와 출력 구조가 Genesis 실험 환경에 맞춰져 있습니다.

주요 특징:

- `scene_generator.py`가 cluttered scene을 생성합니다.
- `object_occlusion.py`가 target occlusion 데이터를 생성합니다.
- `similarity_map_generator.py`가 target category와 scene object category 유사도를 segmentation image에 반영합니다.
- `distribution_map.py`가 segmentation/depth 결과를 결합해 후처리 map을 생성합니다.

## 주요 입력과 산출물

### 입력

- USD/USDC object asset
- target object 이름 또는 target folder 이름
- 시뮬레이터별 camera, workspace, object placement parameter
- 학습 시 사용할 image/label pair

### 산출물

일반적으로 각 backend의 `output/<target_name>/` 아래에 다음과 같은 데이터가 생성됩니다.

- `scene/rgb`: cluttered scene RGB image
- `scene/depth`: scene depth array 또는 image
- `scene/seg`: semantic segmentation image와 mapping metadata
- `target/rgb`: target-only 또는 target scan RGB image
- `target/depth`: target depth data
- `target/seg`: target segmentation image
- `similarity_map`: target과 scene object category 유사도 기반 map
- `distribution_map`: target 위치 추정에 사용할 최종/중간 distribution map

## 사용 시 주의사항

- Isaac Sim 관련 스크립트는 `SimulationApp` 초기화 이후에만 Isaac/Omniverse 모듈 import가 가능하도록 작성되어 있습니다.
- GPU 메모리 사용량은 scene 수, 해상도, batch size에 민감합니다. `distribution_map_GPU.py`의 batch parameter를 환경에 맞게 조정하세요.
- Asset 경로는 코드 내부 상수에 의존하는 부분이 있으므로, 새 asset을 추가할 때는 기존 디렉토리 convention을 유지하는 것이 좋습니다.
- 생성 데이터는 용량이 커질 수 있으므로 git에 직접 추가하기보다 별도 storage 또는 ignored output directory로 관리하는 것을 권장합니다.
