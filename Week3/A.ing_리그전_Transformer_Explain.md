# Aing_리그전 Transformer Fine-tune Explain

## 리그전 목적
- 제한 시간: **120분**
- 목표: **Fine-tune 이후** 번역 모델의 **Multi30k (de→en) test SacreBLEU(BLEU)** 점수를 최대화합니다.
- 리그전 방식: 이미 구현된 코드/라이브러리 위에서 **하이퍼파라미터 튜닝**을 통해 점수를 올립니다.

## 점수(리더보드) 기준
- **FINAL_TEST_BLEU = Fine-tune 이후 Multi30k test SacreBLEU(BLEU)**
- 노트북 마지막 부분에서 `FINAL_TEST_BLEU`가 로그로 출력됩니다.

## 튜닝 허용 범위(중요)
이번 리그전에서 바꿀 수 있는 값은 **딱 2종류**입니다.

1) **Model(구조) 하이퍼파라미터**  
2) **Fine-tune 단계의 Optimizer / Schedule 하이퍼파라미터**

그 외(Tokenizer/BPE, MAX_LEN, batch size, 데이터 샘플링, Pre-train 학습 설정, 디코딩 설정 등)는 **고정**입니다.  
시작 설정은 일부러 보수적으로(성능이 조금 낮게) 설정해서 간단한 튜닝만으로도 점수를 올릴 수 있게 설계했습니다.

---

## Flow Map
1) Setup: 설치/임포트/시드/디바이스  
2) Datasets: Multi30k(기본학습/평가) + IWSLT(파인튜닝) 로드  
3) Tokenizer: Multi30k train으로 BPE 학습(고정)  
4) Model: from-scratch encoder-decoder Transformer 생성 (**🔧 TUNE 가능: 구조만**)  
5) Stage A: Pre-train (Multi30k, **학습 하이퍼파라미터 고정**)  
6) Stage B: Fine-tune (IWSLT, **🔧 TUNE 가능: optimizer/schedule**)  
7) Evaluate: Multi30k test BLEU → `FINAL_TEST_BLEU`

---

## 🔧 TUNE: 리그전에서 바꿀 하이퍼파라미터(딱 2종류)

코드에서 **🔧 `TUNE`** 로 표시된 값만 수정합니다.  
이 노트북에서는 **Model(구조)** 과 **Fine-tune Optimizer/Schedule**만 튜닝하도록 구성했습니다.

### A) Model(구조) 하이퍼파라미터
- `d_model`: 모델 차원(임베딩/hidden 폭)
- `nhead`: head 개수 (`d_model % nhead == 0` 필수)
- `num_layers`: encoder/decoder 레이어 수(동일 개수로 고정)
- `d_ff`: FFN 중간 차원(보통 `2~4 × d_model` 범위에서 시작)
- `dropout`: 드롭아웃 비율(과적합 방지)

### B) Fine-tune Optimizer / Schedule 하이퍼파라미터
- `learning_rate`: fine-tune 학습률
- `lr_scheduler_type`: 스케줄 타입(예: `linear`, `cosine`, `inverse_sqrt`)
- `warmup_steps`: warmup 스텝 수
- `weight_decay`: 가중치 감쇠(일반화 도움)
- `label_smoothing_factor`: 라벨 스무딩(너무 크면 학습이 둔해질 수 있음)

### 고정(변경하지 않음)
- Tokenizer/BPE, `VOCAB_SIZE`, `MAX_LEN`
- batch size/accumulation, 데이터 샘플링(Subset 크기)
- Pre-train 단계의 학습 하이퍼파라미터(steps, LR, warmup 등)
- 디코딩 설정(beam 등) 및 평가 방식

---

## 데이터셋 정리

### Multi30k (de→en)
- **용도**
  - Pre-train(기본 학습) 데이터
  - Fine-tune 이후 **최종 점수 평가(test BLEU)** 데이터
- **특징**
  - 이미지 캡션 중심이라 비교적 **짧은 문장**이 많습니다.
- **이 노트북에서의 사용 규칙(고정)**
  - `train`: 속도를 위해 **부분 샘플(subset)** 사용 (기본: 20,000쌍)
  - `validation/test`: **전체 사용**
- **리그전에서의 역할**
  - 리더보드 점수는 **Fine-tune 이후 Multi30k test BLEU** 입니다.
  - 따라서 Fine-tune은 IWSLT로 하더라도, **Multi30k 성능을 올리는 설정**이 중요합니다.

### IWSLT 2017 (de→en)
- **용도:** Fine-tune(미세조정) 데이터
- **특징**
  - TED talk 번역(구어체/긴 문장/다양한 주제) 성격이 있어 Multi30k와 **문체/길이/어휘 분포가 다를 수** 있습니다.
- **이 노트북에서의 사용 규칙(고정)**
  - `train`: 속도를 위해 **부분 샘플(subset)** 사용 (기본: 50,000쌍)
  - (주의) Colab의 `datasets>=4` 환경에서는 IWSLT 로더가 *dataset script(.py)* 방식이면 막힐 수 있어,
    본 노트북은 **parquet 파일을 직접 로드**하도록 구성되어 있습니다.
- **리그전에서의 역할**
  - IWSLT로 fine-tune 하면서도, 최종 점수는 Multi30k에서 측정되므로  
    “IWSLT에만 과적합”하지 않게 **Optimizer/Schedule**을 신중히 고르는 것이 핵심입니다.

---

## 우리가 튜닝하는 하이퍼파라미터 자세히 (바꿀 수 있는 것만)

리그전에서 바꿀 수 있는 값은 **딱 2종류**입니다.

1) **Model(구조) 하이퍼파라미터**: 모델의 “표현력/용량”을 결정  
2) **Fine-tune Optimizer / Schedule 하이퍼파라미터**: fine-tune에서 “얼마나/어떻게” 업데이트할지 결정

그 외(Tokenizer/BPE, MAX_LEN, batch size, 데이터 샘플링, Pre-train 설정, 디코딩 설정 등)는 **고정**입니다.

---

### A) Model(구조) 하이퍼파라미터 (MODEL_HP)

#### 1) `d_model` (모델 차원, hidden size)
- **의미**: 토큰이 흐르는 벡터의 폭(표현력의 기본 크기). 임베딩/어텐션/FFN의 기준이 되는 핵심 축.
- **값을 키우면(↑) 기대 효과**
  - 표현력이 늘어 **BLEU가 오를 가능성**이 커짐
  - attention head마다 사용할 수 있는 차원도 커져(동일 nhead 기준) 안정적인 경우가 많음
- **값을 줄이면(↓) 기대 효과**
  - 속도↑, 메모리↓ → 120분 동안 실험 횟수↑
  - 하지만 너무 작으면 **언더피팅(학습이 덜 됨)**으로 BLEU가 잘 안 오를 수 있음
- **주의**
  - `d_model`을 키우면 같은 step budget에서도 학습이 느려질 수 있어 “단순히 크게”가 항상 정답은 아닙니다.

#### 2) `nhead` (Multi-Head Attention head 개수)
- **의미**: Attention을 여러 관점으로 병렬 계산(서로 다른 “관계”를 동시에 보게 함).
- **제약(필수)**: `d_model % nhead == 0`
  - head 당 차원: `d_k = d_model / nhead`
- **값을 키우면(↑) 기대 효과**
  - 서로 다른 패턴을 보는 head가 늘어 **표현 다양성**이 증가할 수 있음
- **값을 줄이면(↓) 기대 효과**
  - head 당 차원(d_k)이 커져 **각 head가 더 풍부한 표현**을 가질 수 있음
- **흔한 실패**
  - nhead를 너무 키우면 d_k가 너무 작아져 **성능이 떨어질** 수 있음(논문에서도 너무 많은 head는 성능이 떨어지는 경향을 관찰).

#### 3) `num_layers` (encoder/decoder 레이어 수)
- **의미**: 모델 깊이(Transformer 블록을 몇 번 쌓는지).
- **값을 키우면(↑) 기대 효과**
  - 더 깊은 변환이 가능해져 **성능 잠재력↑**
- **값을 줄이면(↓) 기대 효과**
  - 속도↑, 안정성↑(특히 작은 데이터/짧은 step budget에서)
- **주의**
  - 깊이가 늘면 최적화가 어려워질 수 있어 warmup/learning rate 조합의 중요도가 커질 수 있음.

#### 4) `d_ff` (FFN 중간 차원)
- **의미**: 각 Transformer 블록의 FFN 폭(비선형 변환 용량).
- **값을 키우면(↑) 기대 효과**
  - 어텐션 뒤에 오는 비선형 변환 능력이 증가 → **표현력↑**
- **값을 줄이면(↓) 기대 효과**
  - 속도↑, 메모리↓ → 실험 횟수↑
- **시작 팁**
  - 보통 `2~4 × d_model`에서 시작합니다. 예) d_model=256이면 d_ff=512~1024.

#### 5) `dropout`
- **의미**: 과적합 방지(Regularization).
- **값을 키우면(↑) 기대 효과**
  - 일반화(검증/테스트) 성능이 좋아질 수 있음
- **값을 줄이면(↓) 기대 효과**
  - 학습은 빨리 되지만 과적합 위험↑
- **흔한 실패**
  - dropout이 너무 크면 loss가 줄지 않고 BLEU가 안 오르는 경우가 생깁니다.
  - 120분 리그전에서는 0.1~0.3 범위가 현실적입니다.

---

### B) Fine-tune Optimizer / Schedule 하이퍼파라미터 (FT_HP)

fine-tune은 **“이미 학습된 모델을 얼마나 강하게 바꿀지”**를 조절하는 단계입니다.  
여기서 하이퍼파라미터를 잘못 잡으면:
- 너무 크게 바꾸면: 기존 지식을 잃는 **망각(catastrophic forgetting)** 위험
- 너무 약하게 바꾸면: fine-tune 효과가 거의 없음

#### 1) `learning_rate`
- **의미**: 업데이트 크기(가장 영향 큰 튜닝 변수)
- **값을 키우면(↑) 기대 효과**
  - 빠르게 적응할 수 있음(초반 BLEU 상승이 빠를 수 있음)
- **값을 줄이면(↓) 기대 효과**
  - 안정적으로 천천히 개선(망각 위험↓)
- **흔한 실패**
  - 너무 크면 BLEU가 출렁이거나 오히려 떨어질 수 있음
  - 너무 작으면 500 step 안에 거의 변화가 없을 수 있음

#### 2) `warmup_steps`
- **의미**: 학습 초반에 LR을 천천히 올려 안정화
- **값을 키우면(↑) 기대 효과**
  - 초반 폭주 방지(특히 큰 learning_rate와 조합 시)
- **값을 줄이면(↓) 기대 효과**
  - 빠르게 큰 업데이트를 시작 → 초반 성능 상승이 빠를 수 있지만 불안정해질 수 있음
- **주의**
  - fine-tune step budget이 고정(예: 500)이라면 warmup이 너무 길면 “유효 학습 구간”이 짧아집니다.

#### 3) `lr_scheduler_type`
- **의미**: 학습률 감소 곡선(어떤 모양으로 LR을 줄일지)
- **선택지 감각**
  - `linear`: 일정하게 감소(가장 단순, baseline용)
  - `cosine`: 초반/중반 천천히, 후반 급격히 감소하는 느낌
  - `inverse_sqrt`: Transformer 논문에서 사용한 warmup + inverse_sqrt 계열과 유사한 감각(고정된 warmup 이후 완만히 감소)

#### 4) `weight_decay`
- **의미**: 가중치에 패널티를 줘서 과적합을 줄이는 규제
- **값을 키우면(↑) 기대 효과**
  - 일반화↑(validation/test 개선 가능)
- **값을 줄이면(↓) 기대 효과**
  - 학습이 더 잘 맞지만 과적합 위험↑

#### 5) `label_smoothing_factor`
- **의미**: 정답을 100% 확신하지 않게 만들어 일반화 도움(논문에서도 BLEU 개선에 도움을 준다고 보고)
- **값을 키우면(↑) 기대 효과**
  - 과적합 완화 / BLEU 개선 가능
- **값을 줄이면(↓) 기대 효과**
  - 학습이 “더 확신 있게” 진행되어 빠르게 맞출 수 있지만 과적합 위험↑
- **흔한 실패**
  - 너무 크게 잡으면 학습이 둔해져 점수가 잘 안 오를 수 있음(특히 step이 짧을 때).

---

## 시작 설정(의도적으로 보수적)
기본값은 일부러 **작은 모델 + 보수적인 fine-tune 설정**으로 시작합니다.  
목표는 “처음 점수는 낮지만, 튜닝으로 점수가 잘 오르는” 상태를 만드는 것입니다.

---

## 미니 용어 사전 & Shape 규칙

이 리그전 노트북은 **논문(Attention Is All You Need)** 의 Transformer를 돌려보면서 배우는 목적이라, 코드에 자주 등장하는 용어/shape를 먼저 정리합니다.

### 자주 보는 용어
- **token(토큰)**: 문장을 쪼갠 최소 단위(여기서는 BPE 서브워드 토큰).
- **vocab(어휘집)**: 토큰 ↔ ID 매핑 테이블. `vocab_size`가 크면 표현력↑, 연산/메모리↑.
- **BPE**: Byte Pair Encoding. 희귀 단어를 서브워드로 쪼개 OOV를 줄이는 방식.
- **<pad>/<bos>/<eos>/<unk>**: padding / 문장 시작 / 문장 끝 / 미등록 토큰.
- **logits**: softmax 전 점수. CrossEntropyLoss는 logits를 입력으로 받음.
- **teacher forcing**: 학습 시 디코더 입력으로 정답을 한 칸 shift해서 넣는 기법.
- **BLEU**: 번역 품질을 보는 대표 지표(리그전 스코어로 사용).

### Transformer에서 shape(차원) 규칙 (batch_first 기준)
- `src`: `(N, S)` = (배치크기 N, 소스 길이 S)
- `tgt`: `(N, T)` = (배치크기 N, 타깃 길이 T)
- 임베딩 후: `(N, L, d_model)`
- attention weight(확률): `(N, heads, L_q, L_k)`

### 마스크(mask)
- **padding mask**: `<pad>` 위치는 의미가 없으므로 attention에서 가림.
- **causal mask**: 디코더가 미래 토큰을 보면 '정답을 미리 보는 치팅'이므로 상삼각을 가림.

### 120분 리그전 팁(중요)
- 시간을 많이 잡아먹는 축: **(1) MAX_LEN, (2) d_model/num_layers, (3) max_steps, (4) vocab_size**
- 추천 패턴: `짧게 여러 번(trial)` → 상위 설정만 `길게 한 번(final)`

### (추가) 논문 밖/헷갈리기 쉬운 개념
- **warmup_steps (웜업 스텝)**: 학습 초반에 학습률(LR)을 아주 작게 시작해 **선형으로 천천히 키우는 구간**. 초반에 파라미터가 랜덤일 때 큰 LR로 폭주하는 것을 막아 안정적으로 학습을 시작하게 해줍니다. Transformer 원 논문은 식 (3)에서 `warmup_steps`를 사용합니다(기본 4000).
- **LR scheduler(학습률 스케줄러)**: step이 진행되면서 LR을 바꾸는 규칙. Transformer 논문에서는 'Noam schedule'(inverse sqrt + warmup)을 사용합니다.
- **gradient clipping(그래디언트 클리핑)**: 기울기(gradient)가 너무 커져 폭주하는 것을 막기 위해, norm을 일정 값 이하로 잘라내는 기법.
- **mixed precision / AMP**: FP16(bfloat16)로 계산해 속도/메모리를 개선하는 기법. 수치 불안정이 있으면 loss scale을 사용합니다.
- **beam search(빔 서치)**: 번역 생성 시 여러 후보 경로를 동시에 탐색해 더 좋은 문장을 찾는 디코딩 방법(빔이 클수록 품질↑/속도↓).

---

##  제작 정보 & 출처

- 제작: 가천대학교 인공지능 학술 동아리 **Aing (A.ing)**

### 사용/참고 자료
- Vaswani et al., **Attention Is All You Need**, NeurIPS 2017.
- A.ing 내부 스터디 자료: *Attention 치트시트*, *Transformer CookBook*.
- HuggingFace Transformers/Datasets 문서: Seq2Seq 학습(Trainer), Multi30k, IWSLT 로딩.
- `tokenizers`(BPE), `sacrebleu`(BLEU 평가) 라이브러리.
