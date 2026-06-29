# 디렉토리 구조 설명

## 최상위 구조

```text
.
├── README.md
├── docs/
│   ├── project-overview.md
│   ├── directory-structure.md
│   └── data-pipeline.md
└── src/
    ├── genesis/
    └── isaac/
```

## `docs/`

프로젝트 설명 문서를 보관하는 최상위 문서 폴더입니다.

| 파일 | 설명 |
| --- | --- |
| `project-overview.md` | 프로젝트 목적, 핵심 개념, backend별 구현 범위, 산출물 설명 |
| `directory-structure.md` | repository의 폴더와 주요 source file 역할 설명 |
| `data-pipeline.md` | 데이터 생성, map 후처리, 모델 학습 순서와 예시 명령 정리 |

## `src/isaac/`

Isaac Sim 기반 합성 데이터 생성과 학습 파이프라인입니다.

| 파일 | 역할 |
| --- | --- |
| `README.md` | Isaac Sim backend의 기존 간단 설명 |
| `vectorized_scene.py` | 병렬 Isaac Sim environment를 만들고 target/category 유사도 기반 scene data를 생성 |
| `object_spawner.py` | USD/USDC asset 탐색, stage loading, workspace spawning을 담당하는 재사용 class |
| `object_occlusion.py` | 단일 환경에서 target object를 여러 위치/회전/높이로 스캔하여 occlusion dataset 생성 |
| `vectorized_object_occlusion.py` | 여러 z-level 환경을 병렬화해 occlusion dataset 생성 속도를 높인 버전 |
| `similarity_map_generator.py` | scene segmentation 결과에 category similarity score를 투영하여 similarity map 생성 |
| `distribution_map_GPU.py` | target/scene depth와 similarity/occlusion 정보를 GPU batch 방식으로 결합해 distribution map 생성 |
| `train_260222.py` | 생성된 training image/label pair로 FCN-ResNet50 또는 DeepLabV3 segmentation model 학습 |

예상되는 보조 디렉토리:

- `asset/`: Isaac Sim에서 참조할 USD/USDC asset 저장 위치
- `output/<target_name>/`: scene, target, similarity map, distribution map 등의 생성 결과 저장 위치
- `train_x/`, `train_y/`, `outputs/`: 학습 input, label, model checkpoint 저장 위치

## `src/genesis/`

Genesis simulator 기반 합성 데이터 생성과 후처리 파이프라인입니다.

| 파일 | 역할 |
| --- | --- |
| `README.md` | Genesis backend의 기존 간단 설명 |
| `scene_generator.py` | target object 기준으로 cluttered scene을 생성하고 RGB/depth/segmentation 저장 |
| `object_occlusion.py` | target-only 또는 target scan 데이터를 생성해 occlusion/depth 분석에 활용 |
| `similarity_map_generator.py` | Genesis asset category similarity를 이용해 scene segmentation을 similarity map으로 변환 |
| `distribution_map.py` | segmentation/depth image를 결합하고 Gaussian blur 등을 적용해 distribution 관련 map 생성 |

예상되는 보조 디렉토리:

- `asset/USD/`: Genesis에서 불러올 object asset 저장 위치
- `output/<target_name>/`: Genesis backend가 생성한 scene/target/map 결과 저장 위치

## Source organization 원칙

- backend별 코드는 `src/isaac`과 `src/genesis`로 분리되어 있습니다.
- 두 backend 모두 `scene 생성 → target occlusion 생성 → similarity map 생성 → distribution map 생성`이라는 유사한 흐름을 따릅니다.
- README는 전체 진입점 역할만 하고, 상세 설명은 `docs/`의 목적별 문서로 이동했습니다.
