# 데이터 생성 및 학습 워크플로

이 문서는 repository에서 의도한 일반적인 실행 순서를 설명합니다. 실제 명령은 Isaac Sim, Genesis, asset 경로, Python environment 구성에 따라 조정해야 합니다.

## 1. Backend 선택

### Isaac Sim

Isaac Sim backend는 NVIDIA Isaac Sim/Omniverse runtime이 필요합니다. `src/isaac`로 이동한 뒤 Isaac Sim이 제공하는 Python launcher 또는 프로젝트 환경의 Python으로 실행합니다.

```bash
cd src/isaac
python vectorized_scene.py --target book_1 --num_scenes 10 --num_envs 4 --headless
```

### Genesis

Genesis backend는 Genesis simulator와 관련 Python dependency가 필요합니다.

```bash
cd src/genesis
python scene_generator.py --target_name pen_1 --num_images 10
```

## 2. Scene dataset 생성

Scene dataset은 여러 물체가 workspace에 배치된 관측 이미지입니다. 일반적으로 다음 데이터가 저장됩니다.

- RGB image
- Depth image 또는 depth array
- Semantic segmentation image
- Segmentation index/mapping metadata

Isaac Sim에서는 `vectorized_scene.py`, Genesis에서는 `scene_generator.py`가 이 단계를 담당합니다.

## 3. Target occlusion dataset 생성

Target object를 다양한 위치, 회전, 높이에 놓고 관측하여 depth/segmentation 패턴을 수집합니다.

Isaac Sim 예시:

```bash
cd src/isaac
python object_occlusion.py --target_name book_1 --headless
# 또는 병렬 z-level 버전
python vectorized_object_occlusion.py --target_name book_1 --headless
```

Genesis 예시:

```bash
cd src/genesis
python object_occlusion.py --target_name pen_1
```

## 4. Similarity map 생성

Similarity map은 target category와 scene에 등장한 object category 간 관계를 점수화한 map입니다. 예를 들어 같은 category는 높은 점수, 연관 category는 중간 점수, 관련이 낮은 category는 낮은 점수를 부여합니다.

Isaac Sim 예시:

```bash
cd src/isaac
python similarity_map_generator.py --target_name book_1
```

Genesis 예시:

```bash
cd src/genesis
python similarity_map_generator.py --target_name pen_1
```

## 5. Distribution map 생성

Distribution map은 target이 있을 가능성이 높은 위치를 나타내는 최종 후처리 결과입니다. 이 단계에서는 보통 다음 정보를 결합합니다.

- target/scene segmentation
- target/scene depth
- target occlusion score
- category similarity score
- visibility threshold 및 blur/normalization parameter

Isaac Sim GPU 버전 예시:

```bash
cd src/isaac
python distribution_map_GPU.py --target_name book_1
```

Genesis 예시:

```bash
cd src/genesis
python distribution_map.py --target_name pen_1 --save
```

## 6. Segmentation model 학습

학습 데이터가 `train_x`, `train_y` 형태로 준비되면 Isaac Sim 쪽 training script를 사용할 수 있습니다.

```bash
cd src/isaac
python train_260222.py
```

학습 script는 기본적으로 FCN-ResNet50 또는 DeepLabV3-ResNet50 기반 segmentation model을 사용하며, class list와 data path는 `Config` class에서 조정합니다.

## 권장 검증 순서

1. `--list_objects` 옵션이 있는 Isaac Sim script로 asset 탐색이 정상 동작하는지 확인합니다.
2. 작은 `--num_scenes` 또는 `--num_images` 값으로 scene generation을 먼저 테스트합니다.
3. target occlusion output의 `rgb`, `depth`, `seg` 개수가 서로 맞는지 확인합니다.
4. similarity map과 distribution map 생성 후 image dimension과 value range를 확인합니다.
5. training 전에 `train_x`와 `train_y` pair 수 및 class 이름 mapping을 확인합니다.
