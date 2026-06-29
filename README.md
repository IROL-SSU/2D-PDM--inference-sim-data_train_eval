# 2D-PDM Inference / Simulation Data Train-Eval

이 저장소는 2D-PDM(2D Probabilistic Distribution Map) 기반 객체 탐색을 위해 시뮬레이터에서 합성 데이터를 생성하고, 유사도/가림/깊이 정보를 결합한 distribution map을 만들며, 생성된 데이터를 이용해 2D segmentation 모델을 학습하는 실험용 프로젝트입니다.

기존 세부 설명은 README 한 파일에 모두 담기보다 `/docs` 문서로 분리했습니다. 아래 문서들을 순서대로 읽으면 프로젝트 목적, 코드 구조, 실행 흐름을 빠르게 파악할 수 있습니다.

## 문서 바로가기

| 문서 | 내용 |
| --- | --- |
| [프로젝트 전체 설명](docs/project-overview.md) | 프로젝트 목표, 핵심 개념, Isaac Sim/Genesis 구현, 주요 산출물 설명 |
| [디렉토리 구조](docs/directory-structure.md) | 최상위 폴더와 `src/isaac`, `src/genesis` 주요 파일 역할 |
| [데이터 생성 및 학습 워크플로](docs/data-pipeline.md) | scene/target 데이터 생성부터 similarity/distribution map, 학습까지의 권장 흐름 |

## 빠른 시작 가이드

1. 사용할 시뮬레이터 백엔드를 선택합니다.
   - Isaac Sim 기반 파이프라인: `src/isaac/`
   - Genesis 기반 파이프라인: `src/genesis/`
2. 각 백엔드의 asset 디렉토리에 USD/USDC 모델이 준비되어 있는지 확인합니다.
3. target object를 기준으로 scene 데이터와 target occlusion 데이터를 생성합니다.
4. segmentation/depth 결과를 후처리하여 similarity map과 distribution map을 생성합니다.
5. 필요한 경우 `src/isaac/train_260222.py`로 segmentation 모델을 학습합니다.

자세한 명령 예시는 [데이터 생성 및 학습 워크플로](docs/data-pipeline.md)를 참고하세요.
